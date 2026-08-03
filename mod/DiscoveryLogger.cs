// mod/DiscoveryLogger.cs
using System.Collections.Generic;
using Modding;
using UnityEngine;

namespace HKRLBot
{
    // Boss-discovery instrumentation for adding a NEW boss to the
    // registries: while enabled it watches whatever fight the human is
    // playing and logs, as "DISCOVERY ..." ModLog lines, what a new
    // registry entry needs -- boss-candidate GameObject names with their
    // HP as each HealthManager first appears, every FSM state transition
    // on objects carrying a HealthManager (bosses name their main FSM
    // freely -- Hornet's is "Control", others differ), the knight's
    // per-scene X extremes / grounded floor Y / max height (tag both
    // arena walls once mid-fight), and the knight X at the instant the
    // statue challenge menu opens. Toggled with F4 (OverlayUI); OFF by
    // default so training runs pay one bool check and ModLog stays quiet.
    // scripts/parse_discovery.py reduces the lines to registry values --
    // its regexes are the format contract for every Log call here.
    public static class DiscoveryLogger
    {
        public static bool Enabled;

        // FindObjectsOfType is a scene sweep; 4/sec is plenty for state
        // transitions (boss states last many frames) and invisible next
        // to a human-played session.
        private const float ScanPeriodSeconds = 0.25f;
        private static float nextScanTime;

        // Last state per FSM component instance id, so each transition
        // logs exactly once.
        private static readonly Dictionary<int, string> lastState = new Dictionary<int, string>();
        // HealthManager instance ids already reported as candidates.
        private static readonly HashSet<int> seenHm = new HashSet<int>();

        private static string lastScene = "";
        private static float minKx, maxKx, floorY, maxKy;
        private static bool menuWasOpen;

        public static void Toggle()
        {
            Enabled = !Enabled;
            HKRLBotMod.Instance.Log($"DISCOVERY logging {(Enabled ? "ON" : "OFF")}");
            if (Enabled)
            {
                lastState.Clear();
                seenHm.Clear();
                lastScene = "";
                menuWasOpen = false;
                ResetArena();
            }
        }

        private static void ResetArena()
        {
            minKx = float.PositiveInfinity;
            maxKx = float.NegativeInfinity;
            floorY = float.PositiveInfinity;
            maxKy = float.NegativeInfinity;
        }

        private static string F(float v) =>
            float.IsInfinity(v) ? "NaN" : v.ToString("F2");

        public static void Tick()
        {
            if (!Enabled || Time.unscaledTime < nextScanTime) return;
            nextScanTime = Time.unscaledTime + ScanPeriodSeconds;
            var mod = HKRLBotMod.Instance;
            string scene = GameManager.instance != null ? GameManager.instance.sceneName : "";
            // Knight extremes are per scene: a workshop stroll must not
            // widen the arena's measured range.
            if (scene != lastScene) { lastScene = scene; ResetArena(); }

            foreach (var hm in Object.FindObjectsOfType<HealthManager>())
            {
                if (seenHm.Add(hm.GetInstanceID()))
                {
                    int hp = ReflectionHelper.GetField<HealthManager, int>(hm, "hp");
                    mod.Log($"DISCOVERY candidate go='{hm.gameObject.name}' hp={hp} scene={scene}");
                }
                foreach (var fsm in hm.gameObject.GetComponents<PlayMakerFSM>())
                {
                    int id = fsm.GetInstanceID();
                    string cur = fsm.ActiveStateName;
                    if (!lastState.TryGetValue(id, out string prev) || prev != cur)
                    {
                        lastState[id] = cur;
                        mod.Log($"DISCOVERY state go='{fsm.gameObject.name}' fsm='{fsm.FsmName}' state='{cur}'");
                    }
                }
            }

            var k = mod.Reader.ReadKnight();
            if (k != null)
            {
                bool grew = false;
                if (k.X < minKx) { minKx = k.X; grew = true; }
                if (k.X > maxKx) { maxKx = k.X; grew = true; }
                if (k.OnGround && k.Y < floorY) { floorY = k.Y; grew = true; }
                if (k.Y > maxKy) { maxKy = k.Y; grew = true; }
                if (grew)
                    mod.Log($"DISCOVERY arena scene={scene} kxRange=[{F(minKx)}, {F(maxKx)}] "
                        + $"floorY={F(floorY)} maxKy={F(maxKy)}");

                bool menuOpen = mod.Reader.IsChallengeMenuOpen();
                if (menuOpen && !menuWasOpen)
                    mod.Log($"DISCOVERY statue knightX={k.X:F2} scene={scene}");
                menuWasOpen = menuOpen;
            }
        }
    }
}
