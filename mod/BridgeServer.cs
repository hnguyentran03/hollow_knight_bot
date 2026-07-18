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
            listener = new TcpListener(IPAddress.Loopback, 9020);
            listener.Start();
            new Thread(AcceptLoop) { IsBackground = true }.Start();
            HKRLBotMod.Instance.Log("Bridge listening on 127.0.0.1:9020");
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
                    stream.ReadTimeout = 10000;
                    reader = new StreamReader(stream);
                    writer = new StreamWriter(stream) { AutoFlush = true };
                    writer.WriteLine("{\"type\":\"hello\",\"version\":1}");
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
            // The write itself is fast and non-blocking (a few hundred bytes into a
            // loopback socket send buffer), so it is safe to do it under the lock --
            // unlike ReadMessage below, there is no long blocking call here that would
            // stall the accept thread. If the peer has gone away (broken pipe / reset)
            // this throws IOException; treat that the same as a read failure so a dead
            // writer can never surface as an unhandled exception on the Unity main
            // thread. See DEVIATION note on ReadMessage for why this matters.
            lock (gate)
            {
                try { writer?.WriteLine(text); }
                catch (IOException) { DropLocked(); }
            }
        }

        // DEVIATION from the brief's listing: the brief's ReadMessage held `gate` for
        // the entire blocking reader.ReadLine() call (up to the 10s ReadTimeout). That
        // was empirically confirmed (via a standalone harness driving this exact file,
        // not the game) to cause two bugs, both only visible under the reconnect race
        // the task brief specifically asked to check for:
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
        //      milliseconds after it connected. This was reproduced directly: the
        //      reconnected client's hello sent fine, but the server's `Connected`
        //      getter never observed true and the reconnected client's own message
        //      was lost (read as EOF) because `reader` had already been nulled out
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
