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
        private int resetGraceFrames;   // let scene/HP settle after reload
        // Sticky "boss confirmed dead" flag for the current episode. See the
        // DEVIATION note above ComputeDoneAndWon for why this exists.
        private bool wonLatched;

        private string Scene => GameManager.instance != null
            ? GameManager.instance.sceneName : "";

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
                if ((string)msg["type"] == "reset") awaitingReset = true;
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
                        holdFramesLeft = FrameSkip;
                        break;
                    case "reset":
                        episodeActive = false;
                        awaitingReset = true;
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
        // Also folds in the boss-latch fix documented on ComputeWon below.

        // DEVIATION (small, same spirit as the one above): the brief computed
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

        private void TickReset(BridgeServer server)
        {
            if (resetGraceFrames > 0) { resetGraceFrames--; return; }

            var b = HKRLBotMod.Instance.Reader.ReadBoss();
            var k = HKRLBotMod.Instance.Reader.ReadKnight();
            bool fightLive = Scene == BossScene && b.Present && b.Hp > 0
                             && k != null && k.Hp > 0 && !k.Dead;
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
                server.SendState(k, b, false, false, Scene, attempt);
                return;
            }
            ResetMacro.Tick();          // drive retry prompt / statue macro
            resetGraceFrames = 2;
        }

        private void Awake()
        {
            phase = Phase.AwaitingAction;
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
        private static int t;
        // From mod/DISCOVERED.md section 3 ("Statue-stand X in GG_Workshop"),
        // recorded from a live play session with the F1 overlay on 2026-07-18:
        // "Knight X at Hornet statue in GG_Workshop: 62.21". This is a measured
        // value, not an estimate -- do not change it without re-measuring in
        // game (DISCOVERED.md's own warning: a wrong-but-plausible number here
        // silently corrupts the reset macro with no visible error). If the game
        // build or arena layout ever changes, re-verify against the overlay
        // before trusting this again.
        private const float StatueX = 62.21f;

        public static void Tick()
        {
            var mod = HKRLBotMod.Instance;
            string scene = GameManager.instance != null ? GameManager.instance.sceneName : "";
            t++;
            var b = new ActionButtons();

            if (scene == "GG_Hornet_1")
            {
                // Dead in the boss scene: pulse confirm (jump button) at the retry prompt.
                b.Jump = (t % 40) < 4;
            }
            else if (scene == "GG_Workshop")
            {
                var k = mod.Reader.ReadKnight();
                if (k == null) return;
                if (Mathf.Abs(k.X - StatueX) > 0.5f)
                {
                    b.Left = k.X > StatueX;
                    b.Right = k.X < StatueX;
                }
                else
                {
                    // At the statue: pulse Up to open the challenge menu,
                    // then confirm pulses to select Attuned and begin.
                    b.Up = (t % 60) < 4;
                    b.Jump = (t % 60) >= 30 && (t % 60) < 34;
                }
            }
            mod.Input.Apply(b);
        }
    }
}
