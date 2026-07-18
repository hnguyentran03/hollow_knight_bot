// mod/EpisodeManager.cs
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace HKRLBot
{
    public class EpisodeManager : MonoBehaviour
    {
        private const int FrameSkip = 4;
        private const string BossScene = "GG_Hornet_1";

        // Decision-cycle state machine for an active episode. See the DEVIATION
        // note above LateUpdate for why this replaced the brief's single-phase
        // "send state, then read+apply action" ordering.
        private enum Phase { AwaitingAction, Holding }

        private Phase phase;
        private int holdFramesLeft;
        private int attempt;
        private bool episodeActive;
        private bool awaitingReset;     // trainer asked for reset; waiting for fight to be live
        // Throttles ResetMacro.Tick() to run once every 3 real (rendered) frames
        // instead of every frame -- see the DEVIATION note above TickReset for
        // why, and for how this relates to ResetMacro's own macro-tick constants.
        private int resetGraceFrames;
        // Sticky "boss confirmed dead" flag for the current episode. See the
        // DEVIATION note above ComputeWon for why this exists.
        private bool wonLatched;

        // Final-review fix (F1): true once TickReset has observed at least one
        // NOT-live frame (fight over / wrong scene / boss not yet respawned)
        // since the current reset was requested. Cleared to false every time a
        // "reset" message is accepted (both the idle-poll and mid-episode
        // paths below). fightLive in TickReset requires this flag in addition
        // to the raw live-condition check -- see the DEVIATION note above
        // TickReset for why a raw live-condition check alone is unsound.
        private bool sawNotLiveSinceReset;

        // Final-review fix (F2): macro-tick budget for the reset macro
        // (ResetMacro.Tick, ticked once per resetGraceFrames cycle -- i.e. once
        // per 3 real frames, ~20 macro-ticks/sec). See the DEVIATION note above
        // TickReset for the justification of the exact number.
        //
        // Final-review Issue 1: this MUST stay strictly below the Python
        // trainer's 30s socket timeout (trainer/hkrl/protocol.py's
        // Connection(timeout=30.0)), which is not itself changeable (settled,
        // verified against the real game). If this budget's tick-equivalent
        // seconds reach or exceed 30s, the trainer's own reset() call times
        // out and the client tears down the connection BEFORE the mod ever
        // reaches the budget check in TickReset below -- so the stuck-macro
        // diagnostic that check logs never gets written. A future edit that
        // raises this constant past 30s worth of ticks (600 at this cadence)
        // silently reintroduces that bug.
        private const int ResetMacroBudgetTicks = 450; // 22.5s @ 20 macro-ticks/sec -- must stay below the 30s trainer socket timeout (trainer/hkrl/protocol.py)

        private string Scene => GameManager.instance != null
            ? GameManager.instance.sceneName : "";

        // DEVIATION from the brief's listing: the brief's single-phase loop did
        // [read knight/boss] -> [SendState] -> [if !done, block for the next
        // action] -> [apply it] every FrameSkip'th frame, in that order, reusing
        // whatever action had been applied (or none, right after reset) for the
        // frames leading up to the SendState call. Traced by hand (and confirmed
        // with a throwaway non-Unity simulation of just this control flow, not
        // committed -- see the report) against a client that, per the protocol
        // spec, sends one action and blocks for exactly one state reply each
        // round: with that ordering, the state sent in reply to the client's
        // Nth action always reflects the *(N-1)th* action's held effect, because
        // the mod computes and sends that state BEFORE it has read the Nth action
        // off the socket at all. The effect of the Nth action isn't reported
        // until the state sent in reply to the (N+1)th action. This is a
        // constant one-cycle mislabeling versus the brief's own protocol text
        // two lines above ("action -> mod holds those buttons for FRAME_SKIP
        // frames, then replies with the next state message"): the reply to an
        // action should reflect having held THAT action, not the previous one.
        // Fix: split each decision cycle into two phases -- AwaitingAction (block
        // for the next action, apply it immediately) and Holding (let FrameSkip
        // frames elapse under that action, THEN sample state and send). This
        // matches the spec text exactly and each SendState now reflects exactly
        // the action the client is currently waiting on a reply for.
        //
        // Task 6 review (Important 1): the AwaitingAction frame itself already
        // holds the action (Apply() happens on that frame, via TryApplyButtons
        // in the "action" case below), so it already counts as 1 of the
        // FrameSkip=4 held frames. Only FrameSkip-1 further Holding frames are
        // needed to reach a true total of 4 rendered frames per decision
        // (~15 Hz at 60 fps), matching the plan's stated FRAME_SKIP=4. The
        // original code set holdFramesLeft = FrameSkip (4 further frames on top
        // of the already-held AwaitingAction frame), making the real cycle 5
        // frames (~12 Hz) instead -- see where holdFramesLeft is assigned below.
        // This matters beyond raw rate: Task 7's HKEnv uses max_steps=2700, which
        // is the plan's ~3-minute episode timeout expressed in decision steps and
        // is only correct at 15 Hz; at 12 Hz the same 2700 steps is 3.75 minutes,
        // so the rate error would have silently corrupted the episode timeout too.
        //
        // Also folds in the boss-latch fix documented above ComputeWon.
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
                resetGraceFrames = 0;
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
                // LateUpdate call can freeze the game for up to 10s here, exactly as
                // it already does for BridgeServer's own documented "genuine hang"
                // case (see Task 5 report). This is accepted, not fixed, in this
                // task: the client is expected to send "reset" immediately after
                // connecting / immediately after receiving a done=true state, per
                // the standard Gym reset() pattern, so the common case is sub-second.
                // If a trainer that legitimately needs long idle gaps between
                // episodes is ever built, BridgeServer's ReadTimeout is the thing to
                // revisit, not this loop -- flagged in the report, out of scope here.
                var msg = SafeReadMessage(server);
                if (msg == null) return;
                if ((string)msg["type"] == "reset")
                {
                    awaitingReset = true;
                    sawNotLiveSinceReset = false;   // F1: require a fresh not-live observation
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
                        // FrameSkip - 1: this AwaitingAction frame already held
                        // the action (Apply() just above), so it counts as the
                        // first of FrameSkip held frames -- see the DEVIATION
                        // note above LateUpdate (Important 1).
                        holdFramesLeft = FrameSkip - 1;
                        break;
                    case "reset":
                        episodeActive = false;
                        awaitingReset = true;
                        sawNotLiveSinceReset = false;   // F1: require a fresh not-live observation
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
            // FrameSkip frames before sampling/reporting the resulting state.
            holdFramesLeft--;
            if (holdFramesLeft > 0) return;

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

        // DEVIATION (small, same spirit as the one above LateUpdate): the brief computed
        // `won` as a one-shot `b.Present && b.Hp <= 0` on whatever frame the
        // decision cycle happened to land on. Because state is only sampled once
        // every FrameSkip frames, it's possible for the boss's GameObject to be
        // torn down by its death sequence (Present flips to false) in the same
        // FrameSkip window in which its HP reading crossed to <=0, without this
        // loop ever observing a frame where both were true simultaneously. In
        // that case the original one-shot check reports won=false forever, and
        // the episode would only end later via the `Scene != BossScene` fallback
        // once the game transitions to GG_Workshop -- reporting a real win as a
        // loss/timeout to the trainer, corrupting the reward signal. Fix: latch
        // `wonLatched` the first time we ever observe (Present && Hp<=0) in a
        // given episode, and keep reporting won=true for the rest of that episode
        // even if the boss object later disappears. Reset on every fresh episode
        // start (TickReset's fightLive branch).
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

        // DEVIATION / Task 6 review (Important 2): resetGraceFrames throttles
        // ResetMacro.Tick() to run once every 3 real (rendered) frames, not
        // every frame: each time this method actually reaches the bottom and
        // calls Tick(), it sets resetGraceFrames = 2, which burns the next 2
        // LateUpdate calls (they hit the early-return above without checking
        // fightLive or ticking the macro) before the 3rd call ticks again. This
        // means ResetMacro's own `t` counter advances once per 3 real frames
        // (~50ms at 60 FPS), NOT once per rendered frame -- so its modulo
        // constants are macro-ticks, not frames. Chosen resolution: keep this
        // 3-frame cadence as-is (it is already the tested/reasoned-about
        // behavior; changing the real-time pulse durations right before the
        // human tunes them against the live game would add risk for no
        // benefit) but make the unit unambiguous -- see the named
        // *Ticks constants and comments in ResetMacro below, which spell out
        // both the macro-tick counts and their real-frame/real-time equivalents.
        //
        // Final-review fix (F1): the raw live-condition check below
        // (Scene==BossScene && boss alive && knight alive) is true not only
        // for a freshly (re)started fight but also for a fight that is simply
        // STILL GOING when a "reset" arrives -- e.g. the trainer truncating an
        // episode client-side (HKEnv's max_steps) without the fight actually
        // having ended. Accepting the raw condition there would silently
        // continue the same fight as "episode N+1" (wrong _max_bhp baseline,
        // reward computed against a half-finished fight, `attempt` bumped so
        // nothing downstream notices). Fix: `fightLive` additionally requires
        // `sawNotLiveSinceReset`, which only becomes true once this method has
        // observed a NOT-live frame (fight over, wrong scene, or boss not yet
        // respawned) since the reset was requested -- see the field's comment
        // above. Traced against all three reset origins:
        //   - Death: the knight is already dead the instant the client's
        //     reset arrives (that's why `lost` was true), so the very first
        //     TickReset tick observes not-live and sets the flag immediately
        //     -- unaffected in practice.
        //   - Win: the boss is already dead (or the scene has already started
        //     leaving GG_Hornet_1) the instant the reset arrives, so likewise
        //     the flag is set on the first tick -- unaffected in practice.
        //   - Truncation while the fight is genuinely still live: the flag is
        //     NOT set yet, so fightLive stays false even though the raw
        //     condition is true. ResetMacro.Tick() still runs every grace
        //     cycle (below) and -- because this method already called
        //     Input.Clear() when the "reset" was accepted -- the knight is no
        //     longer being played and only receives the scene's retry-confirm
        //     Jump pulse (ResetMacro's GG_Hornet_1 branch does not
        //     conditionalize on k.Dead, so it runs unconditionally). Against
        //     an aggressive, un-dodged Hornet this reliably lets the fight
        //     actually end (a real death), which flips the raw condition to
        //     not-live, sets the flag, and lets the existing death-retry
        //     macro carry the rest of the way to a genuine fresh fight -- so
        //     "episode N+1" really is a new attempt, not a continuation. This
        //     is the "fight ends" case cited in the review; ResetMacroBudgetTicks
        //     below is the backstop for the (unlikely but possible) case where
        //     the passive knight never gets hit.
        private void TickReset(BridgeServer server)
        {
            if (resetGraceFrames > 0) { resetGraceFrames--; return; }

            var b = HKRLBotMod.Instance.Reader.ReadBoss();
            var k = HKRLBotMod.Instance.Reader.ReadKnight();
            bool live = Scene == BossScene && b.Present && b.Hp > 0
                        && k != null && k.Hp > 0 && !k.Dead;
            if (!live) sawNotLiveSinceReset = true;
            bool fightLive = live && sawNotLiveSinceReset;
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
                // up to FrameSkip frames later.
                HKRLBotMod.Instance.Input.Clear();
                // Final-review fix (F4): log the boss's HP the instant the fight
                // is confirmed live again. Because `live` just flipped
                // false->true this very tick (a fresh (re)spawn -- see
                // sawNotLiveSinceReset above), b.Hp here IS the fight's max HP,
                // before any damage has been dealt. Hornet 1 Attuned has a
                // known/expected HP pool; an Ascended or Radiant tier being
                // accepted by mistake (the statue macro cannot see which tier
                // is highlighted -- see the F4 finding) would show up here as
                // an unexpectedly large number, which is otherwise invisible
                // (the trainer's _max_bhp normalizes it away). This is
                // deliberately just the raw HP reading: no PlayerData field for
                // the selected challenge tier was available to verify, and
                // guessing at one risks logging a confidently wrong value that
                // looks like real diagnostic signal -- see the report.
                HKRLBotMod.Instance.Log(
                    $"Episode {attempt} starting: scene={Scene} bossMaxHp={b.Hp} knightHp={k.Hp}");
                server.SendState(k, b, false, false, Scene, attempt);
                return;
            }

            // Final-review fix (F2, retimed by Issue 1 of the final review):
            // the reset macro has no deadline of its own and, while
            // awaitingReset, this loop never reads the socket (see the
            // DEVIATION note above LateUpdate's idle branch) -- so a stuck
            // macro would otherwise drive virtual buttons in-game forever
            // with no detector. Budget: ResetMacroBudgetTicks (450 macro-ticks
            // == 22.5s @ ~20 macro-ticks/sec, since one macro-tick == 3
            // rendered frames == 50ms @ 60fps). This must be, and is, below
            // the Python trainer's 30s socket timeout (hkrl.protocol.
            // Connection): if it weren't, the trainer's own reset() call
            // times out and the client tears down the connection before the
            // mod ever reaches this check, so the diagnostic log just below
            // -- the entire point of this budget -- would never get written
            // (see the constant's own comment above). At 22.5s the mod gives
            // up, logs where the macro got stuck, and drops the connection
            // with 7.5s of margin to spare before the trainer's own timeout
            // would otherwise fire. Sized against the macro's own real
            // timings: the retry-confirm cycle is RetryPulsePeriodTicks=40
            // ticks (2.0s) and the statue menu/confirm cycle is
            // StatueMenuPeriodTicks=60 ticks (3.0s) -- any legitimate reset
            // (death-retry, or win-path walk-to-statue plus at most a couple
            // of menu cycles) should complete within a handful of those
            // cycles, comfortably inside 22.5s.
            if (ResetMacro.Ticks >= ResetMacroBudgetTicks)
            {
                HKRLBotMod.Instance.Log(
                    $"EpisodeManager: reset macro exceeded its {ResetMacroBudgetTicks}-tick "
                    // Note: ResetMacroBudgetTicks / 20 here must stay a
                    // floating-point division -- integer division silently
                    // truncates 450/20's true 22.5s down to 22s. See Issue 1
                    // of the final review, which caught this while retiming
                    // the budget from 900 (45s) to 450 (22.5s) ticks.
                    + $"(~{ResetMacroBudgetTicks / 20.0:F1}s) budget -- giving up. "
                    + $"Last attempted branch='{ResetMacro.LastBranch}', scene={Scene}, "
                    + $"knightX={(k != null ? k.X.ToString("F2") : "?")}. Clearing input and "
                    + "dropping the connection.");
                HKRLBotMod.Instance.Input.Clear();
                server.Drop();
                awaitingReset = false;
                episodeActive = false;
                resetGraceFrames = 0;
                return;
            }

            ResetMacro.Tick();          // drive retry prompt / statue macro
            resetGraceFrames = 2;
        }
    }

    // Navigates back into the fight using virtual inputs.
    // Death in a HoG fight -> "defeated, retry?" prompt: confirm restarts instantly.
    // Win -> game returns to GG_Workshop: walk to the statue, press up, confirm.
    //
    // Apply()/Input.Apply is only ever invoked from EpisodeManager.LateUpdate (via
    // TickReset -> ResetMacro.Tick, both called synchronously on the Unity main
    // thread by Unity's own MonoBehaviour message loop) or from
    // EpisodeManager.LateUpdate's action-handling branch directly. There is no
    // background thread anywhere in this file, BridgeServer's AcceptLoop thread
    // never touches VirtualInput, and BridgeServer.ReadMessage()'s blocking read
    // (invoked synchronously from LateUpdate) happens on the calling (main)
    // thread too, not on a separate one -- it just blocks that thread. So every
    // Apply() call is on the Unity main thread, which is required: VirtualDevice's
    // State field is written wholesale by Apply() but read field-by-field across
    // seven UpdateWithState calls in VirtualDevice.Update(), which InControl also
    // calls from the main thread. An off-thread Apply() racing that read would
    // produce a torn frame (some old buttons, some new). Verified by inspection:
    // grep for every call site of `.Input.Apply(` in the mod (EpisodeManager.cs,
    // ResetMacro.Tick below, and OverlayUI.cs's F2 debug wiggle) -- all three are
    // MonoBehaviour Update/LateUpdate bodies, i.e. main-thread by construction.
    public static class ResetMacro
    {
        // Task 6 review (Important 2): `t` is driven by EpisodeManager.TickReset,
        // which calls Tick() once every 3 real (rendered) frames (see the
        // DEVIATION note above TickReset) -- so `t` counts MACRO-TICKS, not
        // rendered frames: 1 macro-tick == 3 real frames == ~50ms at 60 FPS. All
        // pulse-timing constants below are named/commented in macro-ticks with
        // their real-frame/real-time equivalents spelled out, since these are
        // the values the human is about to tune against the live game and needs
        // to know the unit of.
        //
        // Task 6 review (Minor): `t` is static and previously never reset
        // between reset attempts, so each attempt resumed mid-cycle from
        // wherever the last attempt left off. Reset() gives each attempt a
        // deterministic start; called by EpisodeManager whenever a fresh
        // "reset" request is accepted (both the idle-poll and mid-episode
        // "reset" message paths).
        private static int t;

        // Final-review fix (F3/F2): exposes the macro-tick counter and the
        // most recently logged branch name so EpisodeManager can (a) enforce
        // ResetMacroBudgetTicks without duplicating a counter, and (b) report
        // which branch the macro was stuck in when that budget expires.
        public static int Ticks => t;
        public static string LastBranch => lastLoggedBranch;

        // Final-review fix (F3): before this, the reset macro logged nothing
        // at all -- the death-retry auto-confirm and the win-path statue walk
        // have never been executed, and the only diagnostic surface was the F1
        // overlay plus the driver's "still waiting on reset()" heartbeat,
        // neither of which can distinguish "walking to the statue" from "stuck
        // in a menu" from "in the wrong scene entirely". Fix: log the scene,
        // knight X, active branch, and macro tick on (a) every branch
        // transition, so a human sees immediately when the macro moves from
        // one phase to the next, and (b) a periodic heartbeat every
        // HeartbeatIntervalTicks ticks so a macro that's stuck WITHOUT
        // transitioning branches (e.g. stalled mid-walk) still produces
        // regular evidence of what it's doing instead of going silent. Chosen
        // interval: HeartbeatIntervalTicks=40 macro-ticks (2.0s) -- the same
        // cadence as RetryPulsePeriodTicks, so a heartbeat lands roughly once
        // per retry-confirm cycle. Logging every tick (20/sec) would be far
        // too noisy for a human watching ModLog.txt live; 2s is frequent
        // enough to see progress without drowning the log.
        private const int HeartbeatIntervalTicks = 40; // ~2.0s @ 20 macro-ticks/sec
        private static string lastLoggedBranch = "";

        public static void Reset() { t = 0; lastLoggedBranch = ""; }

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
        // RetryPulseTicks out of every RetryPulsePeriodTicks macro-ticks.
        private const int RetryPulseTicks = 4;         // ~12 real frames, ~200ms @ 60fps
        private const int RetryPulsePeriodTicks = 40;   // ~120 real frames, ~2.0s @ 60fps

        // Statue challenge-menu macro (GG_Workshop, at statue): pulse Up to open
        // the menu, then pulse Jump partway through the same cycle to confirm.
        // Both pulses are ConfirmPulseTicks long; StatueMenuPeriodTicks is the
        // full cycle, and StatueConfirmOffsetTicks is how far into the cycle the
        // Jump confirm pulse starts.
        private const int StatueMenuPeriodTicks = 60;     // ~180 real frames, ~3.0s @ 60fps
        private const int ConfirmPulseTicks = 4;          // ~12 real frames, ~200ms @ 60fps
        private const int StatueConfirmOffsetTicks = 30;  // ~90 real frames, ~1.5s @ 60fps

        public static void Tick()
        {
            var mod = HKRLBotMod.Instance;
            string scene = GameManager.instance != null ? GameManager.instance.sceneName : "";
            t++;
            var b = new ActionButtons();
            string branch;
            float knightX = float.NaN;

            if (scene == "GG_Hornet_1")
            {
                // Dead in the boss scene: pulse confirm (jump button) at the retry prompt.
                branch = "dead-retry-pulse";
                b.Jump = (t % RetryPulsePeriodTicks) < RetryPulseTicks;
            }
            else if (scene == "GG_Workshop")
            {
                var k = mod.Reader.ReadKnight();
                // Minor fix: previously `if (k == null) return;` skipped Apply()
                // entirely, leaving whatever button a prior tick pressed stuck
                // held indefinitely (every other path through Tick() ends in an
                // Apply()). Now: if the knight isn't readable yet (e.g. mid
                // scene-load), fall through with an all-false ActionButtons,
                // releasing any held button instead of leaving it stuck.
                if (k != null)
                {
                    knightX = k.X;
                    if (Mathf.Abs(k.X - StatueX) > 0.5f)
                    {
                        branch = "walk-to-statue";
                        b.Left = k.X > StatueX;
                        b.Right = k.X < StatueX;
                    }
                    else
                    {
                        // At the statue: pulse Up to open the challenge menu,
                        // then confirm pulses to select Attuned and begin.
                        branch = "statue-menu";
                        b.Up = (t % StatueMenuPeriodTicks) < ConfirmPulseTicks;
                        b.Jump = (t % StatueMenuPeriodTicks) >= StatueConfirmOffsetTicks
                                 && (t % StatueMenuPeriodTicks) < StatueConfirmOffsetTicks + ConfirmPulseTicks;
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

            // F3: log on every branch transition (immediate visibility into
            // what the macro just started doing) plus a periodic heartbeat
            // every HeartbeatIntervalTicks ticks even if the branch hasn't
            // changed (visibility into a macro stuck mid-branch, e.g. stalled
            // mid-walk). See the field/const comments above for why this rate.
            if (branch != lastLoggedBranch || t % HeartbeatIntervalTicks == 0)
            {
                string xStr = float.IsNaN(knightX) ? "?" : knightX.ToString("F2");
                mod.Log($"ResetMacro: tick={t} scene={scene} branch={branch} knightX={xStr}");
                lastLoggedBranch = branch;
            }

            mod.Input.Apply(b);
        }
    }
}
