// mod/EpisodeManager.cs
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace HKRLBot
{
    public class EpisodeManager : MonoBehaviour
    {
        // Frames-at-60fps that one action is meant to span. Only a numerator
        // for ActionHoldSeconds -- nothing counts rendered frames, since the
        // game renders uncapped.
        private const int FrameSkip = 4;
        // How long each action is held: ~66.7ms, a 15 Hz decision rate. HKEnv's
        // max_steps=2700 (env.py) is a ~3-minute episode timeout against this
        // rate; changing it changes that timeout's real-world length.
        private const float ActionHoldSeconds = FrameSkip / 60f;
        private const string BossScene = "GG_Hornet_1";

        // Decision-cycle state machine for an active episode. See the note
        // above LateUpdate on why the cycle is split into two phases instead
        // of a single "send state, then read+apply action" step.
        private enum Phase { AwaitingAction, Holding }

        private Phase phase;
        // Time.unscaledTime deadline at which the current Holding phase ends
        // and the held-for-ActionHoldSeconds action's resulting state is
        // sampled and sent. See the note above LateUpdate.
        private float holdEndTime;
        private int attempt;
        private bool episodeActive;
        private bool awaitingReset;     // trainer asked for reset; waiting for fight to be live
        // Time.unscaledTime deadline for the next ResetMacro.Tick() call --
        // throttles the macro to run at its intended ~20 ticks/sec
        // (ResetMacro.TickIntervalSeconds) regardless of the game's actual
        // render framerate. See the note above TickReset.
        private float nextMacroTickTime;
        // Sticky "boss confirmed dead" flag for the current episode. See the
        // note above ComputeWon for why this exists.
        private bool wonLatched;

        // True once TickReset has observed at least one NOT-live frame (fight
        // over / wrong scene / boss not yet respawned) since the current
        // reset was requested. Cleared to false every time a "reset" message
        // is accepted (both the idle-poll and mid-episode paths below).
        // fightLive in TickReset requires this flag in addition to the raw
        // live-condition check -- see the note above TickReset for why a raw
        // live-condition check alone is unsound.
        private bool sawNotLiveSinceReset;

        // True once a SceneManager.activeSceneChanged event has fired for a
        // fresh entry into BossScene since the current reset was requested.
        // Cleared at the same two "reset accepted" sites as
        // sawNotLiveSinceReset, so the two flags stay in lockstep. Set by
        // OnActiveSceneChanged below. fightLive in TickReset requires this in
        // addition to sawNotLiveSinceReset: only an actual scene reload both
        // recreates the boss's GameObject (StateReader.OnSceneChange) and
        // restores its HP to a fresh max -- see the note above TickReset.
        private bool sawSceneReentrySinceReset;

        // Wall-clock budget for the reset macro. MUST stay strictly below the
        // trainer's 30s socket timeout (Connection(timeout=30.0) in
        // trainer/hkrl/protocol.py): at 30s or above the client times out and
        // tears down the connection before TickReset's budget check runs, so
        // the stuck-macro diagnostic never gets logged.
        private const float ResetMacroBudgetSeconds = 22.5f; // must stay below the 30s trainer socket timeout (trainer/hkrl/protocol.py)

        private string Scene => GameManager.instance != null
            ? GameManager.instance.sceneName : "";

        // Reuses the same SceneManager.activeSceneChanged plumbing
        // HKRLBotMod.Initialize already subscribes StateReader.OnSceneChange
        // to, rather than a second parallel scene-tracking mechanism.
        private void Awake()
        {
            UnityEngine.SceneManagement.SceneManager.activeSceneChanged += OnActiveSceneChanged;
        }

        private void OnActiveSceneChanged(
            UnityEngine.SceneManagement.Scene from, UnityEngine.SceneManagement.Scene to)
        {
            // Logged unconditionally: whether the Godhome death-retry reloads
            // the arena scene (rather than resetting it in place) is the sole
            // assumption sawSceneReentrySinceReset rests on. If a retry never
            // produces a BossScene entry here, every death reset stalls until
            // ResetMacroBudgetSeconds expires, and this line is the evidence.
            HKRLBotMod.Instance.Log($"Scene change: '{from.name}' -> '{to.name}'");
            if (to.name == BossScene) sawSceneReentrySinceReset = true;
        }

        // Decision-cycle ordering: a naive single-phase loop that reads
        // state, sends it, then blocks for the next action and applies it --
        // all in that order every decision cycle -- pairs the state sent in
        // reply to the client's Nth action with the (N-1)th action's held
        // effect, because the state is computed and sent BEFORE the Nth
        // action is even read off the socket. The effect of the Nth action
        // wouldn't be reported until the reply to the (N+1)th action -- a
        // constant one-cycle mislabeling that breaks the Gymnasium contract
        // (each step()'s returned state must reflect the action just taken).
        // Fix: split each decision cycle into two phases -- AwaitingAction
        // (block for the next action, apply it immediately) and Holding (let
        // ActionHoldSeconds elapse under that action, THEN sample state and
        // send). This way each SendState reflects exactly the action the
        // client is currently waiting on a reply for.
        //
        // Holding is timed in seconds, not rendered frames: the game renders
        // uncapped (~200fps observed), so a frame count would shrink the hold
        // and inflate the decision rate on a fast machine.
        //
        // Time.unscaledTime, not Time.time: Hollow Knight drops timeScale
        // during hit-pause on nail impacts, so a scaled clock would stretch
        // holds in real time during a flurry of hits -- and would stall
        // outright if timeScale ever reached zero. Unscaled time also keeps
        // the reset-macro budget on the same clock as the client's socket
        // timeout.
        //
        // Also folds in the boss-latch handling documented above ComputeWon.
        private void LateUpdate()
        {
            var server = HKRLBotMod.Instance.Server;
            if (!server.Connected)
            {
                // A dropped connection (clean close, timeout, or a malformed-JSON
                // protocol violation we chose to treat as fatal -- see SafeReadMessage)
                // must not leave a half-finished episode or reset macro running
                // against a client that no longer exists. Reset everything so a
                // fresh connection always starts from a clean idle state and must
                // send its own explicit "reset" -- it never inherits an in-flight
                // macro or a stale "awaiting action" episode from the previous
                // client. Also stop holding whatever buttons were last applied,
                // so the Knight doesn't keep walking/attacking into empty air.
                if (episodeActive || awaitingReset) HKRLBotMod.Instance.Input.Clear();
                episodeActive = false;
                awaitingReset = false;
                nextMacroTickTime = 0f;
                return;
            }

            if (awaitingReset)
            {
                TickReset(server);
                return;
            }

            if (!episodeActive)
            {
                // Idle: poll (non-lockstep) for a reset request each frame. Note this
                // still goes through BridgeServer.ReadMessage(), which blocks on the
                // socket's 10s ReadTimeout if the client sends nothing -- i.e. one
                // LateUpdate call can freeze the game for up to 10s here, same as
                // BridgeServer's documented read-timeout ceiling elsewhere. This is
                // accepted rather than avoided: the client is expected to send
                // "reset" immediately after connecting / immediately after
                // receiving a done=true state, per the standard Gym reset() pattern,
                // so the common case is sub-second. A trainer that legitimately
                // needs long idle gaps between episodes would need BridgeServer's
                // ReadTimeout revisited, not this loop.
                var msg = SafeReadMessage(server);
                if (msg == null) return;
                if ((string)msg["type"] == "reset")
                {
                    awaitingReset = true;
                    sawNotLiveSinceReset = false;      // require a fresh not-live observation before the next reset is accepted
                    sawSceneReentrySinceReset = false; // require a fresh BossScene entry before the next reset is accepted
                    ResetMacro.Reset();
                }
                return;
            }

            // ---- active episode: decision-cycle state machine ----
            if (phase == Phase.AwaitingAction)
            {
                // Blocks (main thread) until the client sends its next message, or
                // times out/disconnects (returns null), or sends malformed JSON
                // (SafeReadMessage turns that into a dropped connection + null too).
                var reply = SafeReadMessage(server);
                if (reply == null)
                {
                    episodeActive = false;
                    HKRLBotMod.Instance.Input.Clear();
                    return;
                }
                switch ((string)reply["type"])
                {
                    case "action":
                        if (!TryApplyButtons(reply["buttons"]))
                        {
                            // Well-formed JSON, wrong shape (missing/non-bool button
                            // fields). Same treatment as malformed JSON: this is a
                            // protocol violation we cannot safely recover from
                            // mid-lockstep (we don't know what the client meant), so
                            // drop the connection rather than apply a partial/garbage
                            // ActionButtons or silently keep holding the previous one.
                            HKRLBotMod.Instance.Log("EpisodeManager: malformed 'action' buttons, dropping connection");
                            server.Drop();
                            episodeActive = false;
                            HKRLBotMod.Instance.Input.Clear();
                            return;
                        }
                        phase = Phase.Holding;
                        // The action was just applied above, on this very frame --
                        // start the hold window from now, so the total time the
                        // action is held before the resulting state is sampled is
                        // exactly ActionHoldSeconds, regardless of the render
                        // framerate. See the note above LateUpdate.
                        holdEndTime = Time.unscaledTime + ActionHoldSeconds;
                        break;
                    case "reset":
                        episodeActive = false;
                        awaitingReset = true;
                        sawNotLiveSinceReset = false;      // require a fresh not-live observation before the next reset is accepted
                        sawSceneReentrySinceReset = false; // require a fresh BossScene entry before the next reset is accepted
                        ResetMacro.Reset();
                        HKRLBotMod.Instance.Input.Clear();
                        break;
                    default:
                        HKRLBotMod.Instance.Log($"EpisodeManager: unexpected message type '{reply["type"]}' while awaiting action, dropping connection");
                        server.Drop();
                        episodeActive = false;
                        HKRLBotMod.Instance.Input.Clear();
                        break;
                }
                return;
            }

            // phase == Phase.Holding: let the just-applied action play out for
            // ActionHoldSeconds before sampling/reporting the resulting state.
            if (Time.unscaledTime < holdEndTime) return;

            var k = HKRLBotMod.Instance.Reader.ReadKnight();
            var b = HKRLBotMod.Instance.Reader.ReadBoss();
            bool won = ComputeWon(b);
            bool lost = k == null || k.Dead || k.Hp <= 0;
            bool done = lost || won || Scene != BossScene;

            server.SendState(k, b, done, won, Scene, attempt);

            if (done)
            {
                episodeActive = false;
                HKRLBotMod.Instance.Input.Clear();
                HKRLBotMod.Instance.Log($"Episode {attempt} done, won={won}");
                return;
            }

            phase = Phase.AwaitingAction;
        }

        // A one-shot `b.Present && b.Hp <= 0` check on whatever frame the
        // decision cycle happened to land on is not enough: because state is
        // only sampled once per hold window, it's possible for the boss's
        // GameObject to be torn down by its death sequence (Present flips to
        // false) in the same window in which its HP reading crossed to <=0,
        // without this loop ever observing a frame where both were true
        // simultaneously. A one-shot check would then report won=false
        // forever, and the episode would only end later via the `Scene !=
        // BossScene` fallback once the game transitions to GG_Workshop --
        // reporting a real win as a loss/timeout to the trainer, corrupting
        // the reward signal. Fix: latch `wonLatched` the first time we ever
        // observe (Present && Hp<=0) in a given episode, and keep reporting
        // won=true for the rest of that episode even if the boss object
        // later disappears. Reset on every fresh episode start (TickReset's
        // fightLive branch).
        private bool ComputeWon(BossState b)
        {
            if (b.Present && b.Hp <= 0) wonLatched = true;
            return wonLatched;
        }

        // ReadMessage() does not catch malformed JSON from the client --
        // JObject.Parse throws JsonReaderException, which would otherwise
        // propagate out of this MonoBehaviour's LateUpdate uncaught. A single
        // garbage line is a protocol violation we cannot safely recover from
        // mid-lockstep (we have no idea whether it was meant to be the action
        // we're blocked on, a reset, or garbage), so it gets the same treatment
        // as a disconnect: drop the connection and let the trainer reconnect and
        // send a fresh reset. This also guarantees ReadMessage's caller only ever
        // has to handle "got a JObject" or "null", never an exception.
        private static JObject SafeReadMessage(BridgeServer server)
        {
            try
            {
                return server.ReadMessage();
            }
            catch (JsonReaderException ex)
            {
                HKRLBotMod.Instance.Log($"EpisodeManager: malformed JSON from client, dropping connection: {ex.Message}");
                server.Drop();
                return null;
            }
        }

        private static bool TryApplyButtons(JToken bt)
        {
            try
            {
                HKRLBotMod.Instance.Input.Apply(new ActionButtons
                {
                    Left = (bool)bt["left"], Right = (bool)bt["right"],
                    Up = (bool)bt["up"], Down = (bool)bt["down"],
                    Jump = (bool)bt["jump"], Attack = (bool)bt["attack"],
                    Dash = (bool)bt["dash"]
                });
                return true;
            }
            catch (System.Exception)
            {
                // Missing "buttons" object, missing individual keys, or non-bool
                // values all land here (JToken's explicit bool cast throws on
                // null/wrong-typed tokens). Caller decides what to do; we do NOT
                // call Input.Apply with a partially-built ActionButtons here.
                return false;
            }
        }

        // ---- reset handling ----

        // nextMacroTickTime throttles ResetMacro.Tick() to ~20 ticks/sec on the
        // wall clock, so the macro's pulse cadence and diagnostic logging hold
        // their real-world rate no matter how fast the game renders. Calls
        // before the deadline early-return without checking fightLive.
        //
        // The raw live-condition check below (Scene==BossScene && boss alive
        // && knight alive) is true not only for a freshly (re)started fight
        // but also for a fight that is simply STILL GOING when a "reset"
        // arrives -- e.g. the trainer truncating an episode client-side
        // (HKEnv's max_steps) without the fight actually having ended -- and
        // also true for a brief window during the death-to-retry transition,
        // where the knight's Hp/Dead fields flip back to "alive" before the
        // arena has actually reloaded and restored the boss's HP to a fresh
        // max. Accepting the raw condition in either case would silently
        // declare "episode N+1" against a fight that never really restarted
        // (wrong _max_bhp baseline, reward computed against a half-finished
        // or stale-HP fight, `attempt` bumped so nothing downstream notices).
        // Fix: `fightLive` additionally requires both `sawNotLiveSinceReset`
        // (a NOT-live frame observed since the reset was requested) and
        // `sawSceneReentrySinceReset` (a genuine BossScene re-entry observed
        // since the reset was requested) -- see both fields' comments above.
        // Only an actual scene reload both recreates the boss's GameObject
        // and restores its HP to a fresh max, so gating on it closes the
        // death-path gap the raw condition and sawNotLiveSinceReset alone
        // leave open. This holds for all three reset origins:
        //   - Death: the knight is already dead the instant the client's
        //     reset arrives (that's why `lost` was true), so the very first
        //     TickReset tick observes not-live and sets sawNotLiveSinceReset
        //     immediately. sawSceneReentrySinceReset lags behind it by the
        //     handful of frames the actual arena reload takes, so fightLive
        //     stays false until the reload completes and the boss's HP
        //     reading is trustworthy -- this is the gap that used to let a
        //     mid-fight HP reading masquerade as a fresh max.
        //   - Win: the boss is already dead (or the scene has already started
        //     leaving GG_Hornet_1) the instant the reset arrives, so
        //     sawNotLiveSinceReset is set on the first tick; the subsequent
        //     walk-to-statue and menu confirm both take well over a tick, and
        //     confirming the challenge itself triggers a GG_Hornet_1 scene
        //     load, setting sawSceneReentrySinceReset before the fight is
        //     live again -- unaffected in practice.
        //   - Truncation while the fight is genuinely still live: neither
        //     flag is set yet, so fightLive stays false even though the raw
        //     condition is true. ResetMacro.Tick() still runs every grace
        //     cycle (below) and -- because this method already called
        //     Input.Clear() when the "reset" was accepted -- the knight is no
        //     longer being played and only receives the scene's retry-confirm
        //     Jump pulse (ResetMacro's GG_Hornet_1 branch does not
        //     conditionalize on k.Dead, so it runs unconditionally). Against
        //     an aggressive, un-dodged Hornet this reliably lets the fight
        //     actually end (a real death), which sets sawNotLiveSinceReset and
        //     lets the existing death-retry macro (including its own scene
        //     reload) carry the rest of the way to a genuine fresh fight,
        //     setting sawSceneReentrySinceReset too -- so "episode N+1" really
        //     is a new attempt, not a continuation. ResetMacroBudgetSeconds
        //     below is the backstop for the (unlikely but possible) case
        //     where the passive knight never gets hit.
        private void TickReset(BridgeServer server)
        {
            if (Time.unscaledTime < nextMacroTickTime) return;

            var b = HKRLBotMod.Instance.Reader.ReadBoss();
            var k = HKRLBotMod.Instance.Reader.ReadKnight();
            bool live = Scene == BossScene && b.Present && b.Hp > 0
                        && k != null && k.Hp > 0 && !k.Dead;
            if (!live) sawNotLiveSinceReset = true;
            bool fightLive = live && sawNotLiveSinceReset && sawSceneReentrySinceReset;
            if (fightLive)
            {
                awaitingReset = false;
                episodeActive = true;
                phase = Phase.AwaitingAction;
                wonLatched = false;
                attempt++;
                // The reset macro (ResetMacro.Tick, below) drives virtual input
                // right up until this frame (confirm pulses, walk-to-statue). If we
                // don't clear it here, whatever button the macro last happened to
                // be pressing (e.g. a Jump confirm pulse) stays held into the start
                // of the new episode until the client's first action overwrites it
                // up to ActionHoldSeconds later.
                HKRLBotMod.Instance.Input.Clear();
                // Log the boss's HP the instant the fight is confirmed live
                // again. Because fightLive required both a genuine not-live
                // observation and a genuine BossScene re-entry since the
                // reset (sawNotLiveSinceReset / sawSceneReentrySinceReset
                // above), this is a fresh (re)spawn's HP, not a stale
                // mid-fight reading, so b.Hp here IS the fight's max HP,
                // before any damage has been dealt. Hornet 1 Attuned has a
                // known/expected HP pool; an
                // Ascended or Radiant tier being accepted by mistake (the
                // statue macro cannot see which tier is highlighted) would
                // show up here as an unexpectedly large number, which is
                // otherwise invisible (the trainer's _max_bhp normalizes it
                // away). This is deliberately just the raw HP reading: no
                // PlayerData field for the selected challenge tier exists to
                // verify against, and guessing at one risks logging a
                // confidently wrong value that looks like real diagnostic
                // signal.
                HKRLBotMod.Instance.Log(
                    $"Episode {attempt} starting: scene={Scene} bossMaxHp={b.Hp} knightHp={k.Hp}");
                server.SendState(k, b, false, false, Scene, attempt);
                return;
            }

            // The reset macro has no deadline of its own and, while
            // awaitingReset, this loop never reads the socket (see the note
            // above LateUpdate's idle branch) -- so a stuck macro would
            // otherwise drive virtual buttons in-game forever with no
            // detector. Budget: ResetMacroBudgetSeconds (22.5s of wall-clock
            // time). This must be, and is, below the Python trainer's 30s
            // socket timeout (hkrl.protocol.Connection): if it weren't, the
            // trainer's own reset() call times out and the client tears down
            // the connection before the mod ever reaches this check, so the
            // diagnostic log just below -- the entire point of this budget --
            // would never get written (see the constant's own comment above).
            // At 22.5s the mod gives up, logs where the macro got stuck, and
            // drops the connection with 7.5s of margin to spare before the
            // trainer's own timeout would otherwise fire. Sized against the
            // macro's own real timings: the retry-confirm cycle is
            // RetryPulsePeriodSeconds=2.0s and the statue menu/confirm cycle
            // is StatueMenuPeriodSeconds=3.0s -- any legitimate reset
            // (death-retry, or win-path walk-to-statue plus at most a couple
            // of menu cycles) should complete within a handful of those
            // cycles, comfortably inside 22.5s.
            if (ResetMacro.ElapsedSeconds >= ResetMacroBudgetSeconds)
            {
                HKRLBotMod.Instance.Log(
                    $"EpisodeManager: reset macro exceeded its {ResetMacroBudgetSeconds:F1}s "
                    + $"budget -- giving up. "
                    + $"Last attempted branch='{ResetMacro.LastBranch}', scene={Scene}, "
                    + $"knightX={(k != null ? k.X.ToString("F2") : "?")}. Clearing input and "
                    + "dropping the connection.");
                HKRLBotMod.Instance.Input.Clear();
                server.Drop();
                awaitingReset = false;
                episodeActive = false;
                nextMacroTickTime = 0f;
                return;
            }

            ResetMacro.Tick();          // drive retry prompt / statue macro
            nextMacroTickTime = Time.unscaledTime + ResetMacro.TickIntervalSeconds;
        }
    }

    // Navigates back into the fight using virtual inputs.
    // Death in a HoG fight -> "defeated, retry?" prompt: confirm restarts instantly.
    // Win -> game returns to GG_Workshop: walk to the statue, press up, confirm.
    //
    // Apply()/Input.Apply must only ever run on the Unity main thread:
    // VirtualDevice's State field is written wholesale by Apply() but read
    // field-by-field across seven UpdateWithState calls in
    // VirtualDevice.Update(), which InControl also calls from the main
    // thread. An off-thread Apply() racing that read would produce a torn
    // frame (some old buttons, some new). Every call site in the mod --
    // EpisodeManager.LateUpdate's action-handling branch, TickReset ->
    // ResetMacro.Tick below, and OverlayUI's F2 debug wiggle -- is a
    // MonoBehaviour Update/LateUpdate body, i.e. main-thread by construction;
    // there is no background thread anywhere that touches VirtualInput, and
    // BridgeServer's AcceptLoop thread never does either.
    public static class ResetMacro
    {
        // All timing below is computed from ElapsedSeconds -- real seconds
        // since Reset() -- rather than a count of Tick() calls, so each pulse
        // holds its intended real-world duration at any render framerate.
        private static float resetStartTime;

        // How often EpisodeManager.TickReset calls Tick(): ~20 times/sec.
        // This is a resolution/logging-rate choice, not a correctness
        // requirement -- the pulse math below is computed from ElapsedSeconds
        // directly, so it stays correct even if a particular Tick() call
        // lands a few milliseconds early or late.
        public const float TickIntervalSeconds = 0.05f; // 20 ticks/sec

        // Seconds elapsed (wall-clock, Time.unscaledTime) since the current
        // reset macro run started. Exposed so EpisodeManager can enforce
        // ResetMacroBudgetSeconds without duplicating a clock.
        public static float ElapsedSeconds => Time.unscaledTime - resetStartTime;

        // Most recently logged branch name, so EpisodeManager can report
        // which branch the macro was stuck in when its budget expires.
        public static string LastBranch => lastLoggedBranch;

        // The reset macro (death-retry auto-confirm, win-path statue walk)
        // drives virtual input with no other diagnostic surface besides the
        // F1 debug overlay and the driver's "still waiting on reset()"
        // heartbeat, neither of which can distinguish "walking to the statue"
        // from "in the menu confirming" from "stuck in the wrong scene
        // entirely". So: log the scene, knight X, active branch, and elapsed
        // time on (a) every branch transition, so a human sees immediately
        // when the macro moves from one phase to the next, and (b) a
        // periodic heartbeat every HeartbeatIntervalSeconds even if the
        // branch hasn't changed, so a macro that's stuck WITHOUT
        // transitioning branches (e.g. stalled mid-walk, or endlessly
        // retrying a menu confirm) still produces regular evidence of what
        // it's doing instead of going silent. Chosen interval:
        // HeartbeatIntervalSeconds=2.0s -- the same cadence as
        // RetryPulsePeriodSeconds, so a heartbeat lands roughly once per
        // retry-confirm cycle. Logging every tick (20/sec) would be far too
        // noisy for a human watching ModLog.txt live; 2s is frequent enough
        // to see progress without drowning the log.
        private const float HeartbeatIntervalSeconds = 2.0f;
        private static string lastLoggedBranch = "";
        private static float lastHeartbeatElapsed;

        // Sticky "the statue challenge menu has been opened" latch. See the
        // note above the GG_Workshop branch in Tick() for why this exists.
        private static bool statueMenuLatched;

        // ElapsedSeconds timestamp at which statueMenuLatched flipped true
        // this reset; -1f means "not yet latched". Lets the statue-menu
        // branch below press Up exactly once for this reset (to open the
        // menu) instead of on every StatueMenuPeriodSeconds cycle -- the menu
        // opens with Attuned already highlighted, so repeated Up presses
        // only navigate away from it.
        private static float statueMenuEnteredAt = -1f;

        // ElapsedSeconds timestamp at which GG_Workshop became the active
        // scene; -1f means "not currently in Workshop". The challenge press is
        // held off until WorkshopSettleSeconds past this, because the scene
        // does not accept input during its load/fade-in and the press is a
        // one-shot -- fired too early it is swallowed and never retried.
        private static float workshopEnteredAt = -1f;

        // How long after entering GG_Workshop to wait before pressing Up.
        // Costs a fixed delay on every statue reset, well inside
        // ResetMacroBudgetSeconds, and buys immunity to the fade-in eating the
        // one press that opens the menu.
        private const float WorkshopSettleSeconds = 1.5f;

        public static void Reset()
        {
            resetStartTime = Time.unscaledTime;
            lastLoggedBranch = "";
            lastHeartbeatElapsed = 0f;
            statueMenuLatched = false;
            statueMenuEnteredAt = -1f;
            workshopEnteredAt = -1f;
        }

        // From mod/DISCOVERED.md section 3 ("Statue-stand X in GG_Workshop"),
        // recorded from a live play session with the F1 overlay on 2026-07-18:
        // "Knight X at Hornet statue in GG_Workshop: 62.21". This is a measured
        // value, not an estimate -- do not change it without re-measuring in
        // game (DISCOVERED.md's own warning: a wrong-but-plausible number here
        // silently corrupts the reset macro with no visible error). If the game
        // build or arena layout ever changes, re-verify against the overlay
        // before trusting this again.
        private const float StatueX = 62.21f;

        // Retry-prompt confirm pulse (GG_Hornet_1, dead): hold Jump for
        // RetryPulseSeconds out of every RetryPulsePeriodSeconds.
        private const float RetryPulseSeconds = 0.2f;
        private const float RetryPulsePeriodSeconds = 2.0f;

        // Statue challenge-menu macro (GG_Workshop, at statue). Game bindings:
        // Up (W) challenges at the statue and opens the difficulty menu; Jump
        // (Space) confirms the highlighted difficulty. So the order is fixed:
        // one Up to challenge, then Jump to confirm -- never the reverse.
        //
        // Up is a one-shot (Attuned is already highlighted when the menu opens,
        // and a second Up navigates away from it). Jump then pulses every
        // StatueMenuPeriodSeconds, retrying the confirm until the fight starts.
        // Both pulses are ConfirmPulseSeconds long; the whole sequence is timed
        // from statueMenuEnteredAt rather than an absolute cycle, so a confirm
        // can never land before the challenge press that opens the menu.
        private const float StatueMenuPeriodSeconds = 3.0f;
        private const float ConfirmPulseSeconds = 0.2f;
        private const float StatueConfirmOffsetSeconds = 1.5f;

        public static void Tick()
        {
            var mod = HKRLBotMod.Instance;
            string scene = GameManager.instance != null ? GameManager.instance.sceneName : "";
            float elapsed = ElapsedSeconds;
            var b = new ActionButtons();
            string branch;
            float knightX = float.NaN;

            // The latch only ever means anything inside GG_Workshop; clear it
            // whenever the scene isn't Workshop so a later re-entry (e.g. a
            // fresh reset that lands back in GG_Hornet_1 first) starts the
            // walk-then-menu sequence over rather than resuming mid-confirm
            // against a menu that no longer exists.
            if (scene != "GG_Workshop")
            {
                statueMenuLatched = false;
                workshopEnteredAt = -1f;
            }
            else if (workshopEnteredAt < 0f)
            {
                workshopEnteredAt = elapsed;
            }

            if (scene == "GG_Hornet_1")
            {
                // Dead in the boss scene: pulse confirm (jump button) at the retry prompt.
                branch = "dead-retry-pulse";
                b.Jump = (elapsed % RetryPulsePeriodSeconds) < RetryPulseSeconds;
            }
            else if (scene == "GG_Workshop")
            {
                var k = mod.Reader.ReadKnight();
                // If the knight isn't readable yet (e.g. mid scene-load), fall
                // through with an all-false ActionButtons instead of returning
                // early, so any button held by a prior tick gets released
                // rather than stuck indefinitely (every other path through
                // Tick() ends in an Apply()).
                if (k != null)
                {
                    knightX = k.X;
                    // Latch onto the statue-menu branch the first time the
                    // knight is within the deadband, and never re-check
                    // position again for the rest of this reset. A live run
                    // showed why: the Up pulse below opens the challenge
                    // menu, which takes over player control -- but the
                    // position check used to be re-evaluated every tick, so
                    // the knight drifting even slightly outside the deadband
                    // (whether from residual momentum or the menu itself)
                    // flipped the branch back to walk-to-statue. That branch
                    // never sends the Jump confirm, so the macro pressed
                    // Left/Right into a menu that no longer reads movement
                    // and never confirmed -- exactly the frozen-at-60.66
                    // stall observed in ModLog.txt. Latching on entry means
                    // once the menu is (assumed) open, the macro commits to
                    // the confirm sequence to completion instead of
                    // re-deriving "am I still at the statue" from a position
                    // reading the menu itself can invalidate. This is chosen
                    // over latching on a detected "input stopped moving the
                    // knight" condition: that would need tracking recent
                    // position deltas against which directional buttons were
                    // held, which is more precise about *why* the knight
                    // stopped responding but also more fragile (a legitimate
                    // brief stall against a wall, or a frame of stale
                    // physics data, would look identical) and adds a second
                    // stateful signal to get wrong. Latching on entry is
                    // simple, matches the observed failure exactly, and is
                    // self-limiting: if the menu somehow doesn't open, the
                    // macro just keeps retrying the Jump confirm every
                    // StatueMenuPeriodSeconds (the Up press only ever fires
                    // once -- see statueMenuEnteredAt above) until
                    // EpisodeManager's ResetMacroBudgetSeconds wall-clock
                    // backstop gives up and logs branch='statue-menu' as the
                    // last attempted branch.
                    bool settled = elapsed - workshopEnteredAt >= WorkshopSettleSeconds;
                    if (!statueMenuLatched && settled && Mathf.Abs(k.X - StatueX) <= 0.5f)
                    {
                        statueMenuLatched = true;
                        statueMenuEnteredAt = elapsed;
                    }

                    if (!statueMenuLatched && !settled)
                    {
                        // Hold everything until the scene finishes loading --
                        // pressing Up now would be swallowed, and walking now
                        // can drift the knight off the statue.
                        branch = "workshop-settling";
                    }
                    else if (!statueMenuLatched)
                    {
                        branch = "walk-to-statue";
                        b.Left = k.X > StatueX;
                        b.Right = k.X < StatueX;
                    }
                    else
                    {
                        // At (or committed to) the statue: challenge once with
                        // Up, then confirm the already-highlighted Attuned
                        // difficulty with Jump, retrying the confirm every
                        // StatueMenuPeriodSeconds if the fight hasn't started.
                        // Both are measured from the branch's own start so the
                        // confirm always trails the challenge.
                        branch = "statue-menu";
                        float sinceMenu = elapsed - statueMenuEnteredAt;
                        b.Up = sinceMenu < ConfirmPulseSeconds;
                        float confirmPos = sinceMenu - StatueConfirmOffsetSeconds;
                        b.Jump = confirmPos >= 0f
                                 && (confirmPos % StatueMenuPeriodSeconds) < ConfirmPulseSeconds;
                    }
                }
                else
                {
                    branch = "workshop-knight-unreadable";
                }
            }
            else
            {
                branch = "unexpected-scene";
            }

            // Log on every branch transition (immediate visibility into what
            // the macro just started doing) plus a periodic heartbeat every
            // HeartbeatIntervalSeconds even if the branch hasn't changed
            // (visibility into a macro stuck mid-branch, e.g. stalled
            // mid-walk, or endlessly retrying a menu confirm that never
            // takes). See the field/const comments above for why this rate.
            // Together with EpisodeManager's own "exceeded budget -- giving
            // up" log, a human reading ModLog.txt can always distinguish
            // walking (branch transitions/heartbeats showing
            // walk-to-statue), confirming (branch transitions/heartbeats
            // showing statue-menu or dead-retry-pulse), and giving up (the
            // budget-exceeded log naming the last attempted branch).
            if (branch != lastLoggedBranch || elapsed - lastHeartbeatElapsed >= HeartbeatIntervalSeconds)
            {
                string xStr = float.IsNaN(knightX) ? "?" : knightX.ToString("F2");
                mod.Log($"ResetMacro: elapsed={elapsed:F2}s scene={scene} branch={branch} knightX={xStr}");
                lastLoggedBranch = branch;
                lastHeartbeatElapsed = elapsed;
            }

            mod.Input.Apply(b);
        }
    }
}
