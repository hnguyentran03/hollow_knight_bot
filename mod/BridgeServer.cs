// mod/BridgeServer.cs
using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace HKRLBot
{
    public class BridgeServer
    {
        private TcpListener listener;
        private TcpClient client;
        private StreamReader reader;
        private StreamWriter writer;
        private readonly object gate = new object();

        public bool Connected { get { lock (gate) return client != null && client.Connected; } }

        public void Start()
        {
            int port = 9020;
            var raw = Environment.GetEnvironmentVariable("HKRL_PORT");
            // Fall back to the default on anything unparseable rather than throwing:
            // a mistyped port must not prevent the game from starting, and the log
            // line below is what reveals which port was actually taken.
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int parsed)
                && parsed > 0 && parsed < 65536)
            {
                port = parsed;
            }
            listener = new TcpListener(IPAddress.Loopback, port);
            listener.Start();
            new Thread(AcceptLoop) { IsBackground = true }.Start();
            HKRLBotMod.Instance.Log($"Bridge listening on 127.0.0.1:{port}");
        }

        private void AcceptLoop()
        {
            while (true)
            {
                var c = listener.AcceptTcpClient();
                lock (gate)
                {
                    DropLocked();
                    client = c;
                    var stream = c.GetStream();
                    // This is a hard ceiling on trainer think-time, not just a
                    // dead-peer detector. ReadMessage's reader.ReadLine() (called
                    // synchronously from EpisodeManager.LateUpdate on the Unity
                    // main thread) throws IOException on timeout, which
                    // DropIfCurrent turns into an unconditional dropped
                    // connection -- NOT a null/"try again" return. So if the
                    // trainer takes longer than 10s to send its next message for
                    // ANY reason (e.g. a PPO policy update running synchronously
                    // between step() calls), the mod drops the connection out
                    // from under it, potentially mid-episode. See the matching
                    // note in trainer/hkrl/env.py. Do not change this value
                    // without also reconsidering that call site.
                    stream.ReadTimeout = 10000;
                    // A write must be bounded for the same reason a read is.
                    // SendState's WriteLine holds `gate` while it writes, and
                    // without a deadline a peer that stops draining its
                    // receive buffer (a wedged/App-Napped trainer) blocks that
                    // write -- and therefore AcceptLoop's ability to install a
                    // reconnecting client -- indefinitely. With WriteTimeout
                    // set, a stalled write instead throws IOException, which
                    // SendState already turns into a clean DropLocked(), so a
                    // wedged peer costs at most this timeout rather than
                    // freezing the bridge forever. Mirrors ReadTimeout's 10s;
                    // see the note in SendState.
                    stream.WriteTimeout = 10000;
                    reader = new StreamReader(stream);
                    writer = new StreamWriter(stream) { AutoFlush = true };
                    writer.WriteLine("{\"type\":\"hello\",\"version\":2}");
                }
                HKRLBotMod.Instance.Log("Trainer connected");
            }
        }

        public void SendState(KnightState k, BossState b, bool done, bool won, string scene, int attempt)
        {
            var msg = new JObject
            {
                ["type"] = "state",
                ["obs"] = new JObject
                {
                    ["kx"] = k?.X ?? 0f, ["ky"] = k?.Y ?? 0f,
                    ["kvx"] = k?.Vx ?? 0f, ["kvy"] = k?.Vy ?? 0f,
                    ["khp"] = k?.Hp ?? 0, ["soul"] = k?.Soul ?? 0,
                    ["on_ground"] = k?.OnGround ?? false, ["dashing"] = k?.Dashing ?? false,
                    ["invuln"] = k?.Invuln ?? false, ["facing_right"] = k?.FacingRight ?? true,
                    ["bx"] = b.X, ["by"] = b.Y, ["bvx"] = b.Vx, ["bvy"] = b.Vy,
                    ["bhp"] = b.Hp, ["boss_state"] = b.FsmState,
                    ["needle_active"] = b.NeedleActive, ["nx"] = b.NeedleX, ["ny"] = b.NeedleY
                },
                ["done"] = done,
                ["info"] = new JObject { ["won"] = won, ["scene"] = scene, ["attempt"] = attempt }
            };
            var text = msg.ToString(Formatting.None);
            // This write is fast in the common case (a few hundred bytes into
            // a loopback socket send buffer). It IS now bounded: AcceptLoop
            // sets stream.WriteTimeout = 10000 on this stream, so if the peer
            // stops draining its receive buffer (e.g. a wedged/blocked
            // trainer) this WriteLine (AutoFlush) blocks at most ~10s rather
            // than forever -- important because it holds `gate` while it
            // writes, and a permanently-stuck write would stall AcceptLoop's
            // ability to accept/install a reconnecting client for as long as
            // it was stuck. When the deadline expires the write throws
            // IOException; a peer that has gone away outright (broken pipe /
            // reset) throws IOException too. Both land in the catch below and
            // are treated the same as a read failure -- a clean DropLocked()
            // -- so neither a wedged nor a dead writer can surface as an
            // unhandled exception on the Unity main thread. See the note on
            // ReadMessage below for the related reconnect-race handling.
            lock (gate)
            {
                try { writer?.WriteLine(text); }
                catch (IOException) { DropLocked(); }
            }
        }

        // Reply to a liveness ping (see EpisodeManager's ping handling).
        // Deliberately a sibling of SendState, using the exact same gated,
        // WriteTimeout-bounded, IOException-drops-cleanly write path: a pong
        // that cannot be written (dead or wedged peer) tears the connection
        // down instead of throwing on the Unity main thread. This is only
        // ever CALLED from EpisodeManager.LateUpdate (the main thread), so a
        // frozen main thread never produces a pong -- which is the entire
        // point of the ping: AcceptLoop's background thread greets a probe
        // even while the main thread is dead, but only a live main thread can
        // answer this.
        public void SendPong()
        {
            lock (gate)
            {
                try { writer?.WriteLine("{\"type\":\"pong\"}"); }
                catch (IOException) { DropLocked(); }
            }
        }

        // Refuse a request with a reason the trainer can surface (e.g. a
        // boss id this build's registry doesn't know). Same bounded, gated
        // write path as SendState/SendPong; the caller decides whether to
        // Drop() afterward.
        public void SendError(string message)
        {
            var msg = new JObject { ["type"] = "error", ["message"] = message };
            var text = msg.ToString(Formatting.None);
            lock (gate)
            {
                try { writer?.WriteLine(text); }
                catch (IOException) { DropLocked(); }
            }
        }

        // ReadMessage must not hold `gate` for the entire blocking
        // reader.ReadLine() call (up to the 10s ReadTimeout), or two bugs
        // follow, both only visible under a reconnect race:
        //   1. AcceptLoop cannot acquire `gate` to install a newly-accepted client
        //      until the in-flight ReadMessage() releases it, so a trainer that
        //      reconnects while a stale ReadMessage() is still blocked on the old,
        //      dead connection is made to wait up to ~10s for its hello -- even
        //      though the new TCP connection was accepted almost instantly.
        //   2. Worse: when the timed-out ReadMessage() unblocks, its `catch
        //      (IOException) { Drop(); }` and AcceptLoop's swap-in of the new client
        //      both race for `gate`. If AcceptLoop wins first (installs the new,
        //      healthy client), the stale ReadMessage()'s unconditional Drop() then
        //      runs and blows away the FIELDS THAT NOW BELONG TO THE NEW CLIENT --
        //      nulling `reader`/`writer`/`client` and closing the brand-new socket
        //      milliseconds after it connected. The observable symptom: the
        //      reconnected client's hello sends fine, but the server's `Connected`
        //      getter never observes true and the reconnected client's own message
        //      is lost (read as EOF) because `reader` has already been nulled out
        //      from under it.
        // Fix: capture the specific StreamReader this call is reading from, do the
        // blocking read OUTSIDE the lock (so AcceptLoop is never stalled by it), and
        // only clear the shared fields afterward if they still refer to that same
        // reader -- i.e. only drop it if nothing has reconnected in the meantime. As
        // a side benefit, closing a superseded connection now unblocks its stale,
        // in-flight ReadLine() promptly instead of leaving it to sit out the full
        // ReadTimeout.
        public JObject ReadMessage()
        {
            StreamReader r;
            lock (gate) { r = reader; }
            if (r == null) return null;
            try
            {
                string line = r.ReadLine();
                if (line == null) { DropIfCurrent(r); return null; }
                return JObject.Parse(line);
            }
            catch (IOException) { DropIfCurrent(r); return null; }
            catch (ObjectDisposedException) { DropIfCurrent(r); return null; }
        }

        private void DropIfCurrent(StreamReader r)
        {
            lock (gate) { if (ReferenceEquals(reader, r)) DropLocked(); }
        }

        public void Drop() { lock (gate) DropLocked(); }

        private void DropLocked()
        {
            reader = null; writer = null;
            client?.Close(); client = null;
        }
    }
}
