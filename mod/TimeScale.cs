// mod/TimeScale.cs
using System;
using UnityEngine;

namespace HKRLBot
{
    // Faster-than-real-time training: run the whole game at k x real time.
    // k comes from the HKRL_TIMESCALE env var (the same launch-env channel
    // as HKRL_PORT), default 1.0, clamped to [Min, Max]. At exactly 1.0
    // Init installs nothing at all -- the k=1 path is byte-for-byte the
    // pre-feature behavior, so existing checkpoints and gates are
    // untouched by this file merely existing.
    public static class TimeScale
    {
        public const float Min = 1f;
        public const float Max = 10f;

        public static float Multiplier { get; private set; } = 1f;

        public static void Init(HKRLBotMod mod)
        {
            Multiplier = ReadMultiplier(mod);
            if (Multiplier == 1f) return;

            // Every game-code write to Time.timeScale funnels through
            // TimeController.SetTimeScaleFactor (verified against the
            // decompiled Assembly-CSharp, 2026-08-22; the only other
            // writers are InControl's test scene and debug tooling, which
            // never run in gameplay). Re-applying the product x Multiplier
            // after every write -- recomputed from the four factor
            // properties, never from the current Time.timeScale -- keeps
            // the hook idempotent (no compounding) and preserves the
            // 0.01 floor, so pause menu and hit-pause still reach a
            // true 0 and stay fully frozen.
            On.TimeController.SetTimeScaleFactor += HookSetTimeScaleFactor;

            // Freeze-frames (nail-hit hit-pause etc.) ramp and hold on
            // UNSCALED time, so at k x their wall duration would be
            // unchanged while decisions shrink to 66.7/k wall-ms -- the
            // bot would see k x more repeated frozen observations per
            // hit than any 1x checkpoint ever saw. Dividing the three
            // duration args by k restores the same decisions-per-freeze
            // at any k. The FreezeMoment(int) preset variant delegates
            // to the float overloads, so these three hooks cover it.
            On.GameManager.FreezeMoment_float_float_float_float +=
                (orig, self, rampDown, wait, rampUp, targetSpeed) =>
                    orig(self, rampDown / Multiplier, wait / Multiplier,
                         rampUp / Multiplier, targetSpeed);
            On.GameManager.FreezeMoment_float_float_float_bool +=
                (orig, self, rampDown, wait, rampUp, runGc) =>
                    orig(self, rampDown / Multiplier, wait / Multiplier,
                         rampUp / Multiplier, runGc);
            On.GameManager.FreezeMomentGC +=
                (orig, self, rampDown, wait, rampUp, targetSpeed) =>
                    orig(self, rampDown / Multiplier, wait / Multiplier,
                         rampUp / Multiplier, targetSpeed);

            // The game only writes Time.timeScale when a factor CHANGES,
            // and all four boot at 1 -- without this one-time apply the
            // multiplier would not take effect until the first pause or
            // hit-pause.
            Apply();
            mod.Log($"TimeScale: running at {Multiplier}x real time");
        }

        private static void HookSetTimeScaleFactor(
            On.TimeController.orig_SetTimeScaleFactor orig,
            ref float field, float val)
        {
            orig(ref field, val);
            Apply();
        }

        private static void Apply()
        {
            float p = TimeController.SlowMotionTimeScale
                    * TimeController.PauseTimeScale
                    * TimeController.PlatformBackgroundTimeScale
                    * TimeController.GenericTimeScale;
            Time.timeScale = p < 0.01f ? 0f : p * Multiplier;
        }

        private static float ReadMultiplier(HKRLBotMod mod)
        {
            var raw = Environment.GetEnvironmentVariable("HKRL_TIMESCALE");
            if (string.IsNullOrEmpty(raw)) return 1f;
            float k;
            if (!float.TryParse(raw, System.Globalization.NumberStyles.Float,
                                System.Globalization.CultureInfo.InvariantCulture,
                                out k))
            {
                mod.Log($"TimeScale: unparseable HKRL_TIMESCALE '{raw}', running at 1x");
                return 1f;
            }
            if (k < Min || k > Max)
            {
                float clamped = Mathf.Clamp(k, Min, Max);
                mod.Log($"TimeScale: HKRL_TIMESCALE {k} outside [{Min}, {Max}], clamped to {clamped}");
                return clamped;
            }
            return k;
        }
    }
}
