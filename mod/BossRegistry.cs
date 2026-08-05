// mod/BossRegistry.cs
using System.Collections.Generic;

namespace HKRLBot
{
    // Mod-side per-boss data, keyed by the boss id the trainer sends in
    // every reset request (protocol v2). The trainer keeps its own registry
    // (trainer/hkrl/bosses.py) holding the data IT consumes (FSM states,
    // arena constants); this one holds only what the mod consumes. Adding a
    // boss means one entry in each, transcribed from an in-game discovery
    // session recorded in DISCOVERED.md.
    public class BossSpec
    {
        public string Id;
        // The Godhome arena scene this boss's fight runs in.
        public string Scene;
        // The boss's root GameObject name, for StateReader's scene scan.
        public string ObjectName;
        // Name of the boss's main FSM on that GameObject -- bosses name it
        // freely (Hornet: "Control", Gruz Mother: "Big Fly Control"), so
        // StateReader locates it per boss.
        public string FsmName;
        // Knight X when standing at this boss's statue in GG_Workshop --
        // a measured value (F1 overlay), like Hornet's in DISCOVERED.md
        // section 3. A wrong-but-plausible number silently corrupts the
        // reset macro; never change without re-measuring.
        public float StatueX;
        // Ceiling for backstop B (wrong-difficulty detection): safely above
        // the boss's measured Attuned max HP, below its next tier's.
        public int MaxAttunedHp;
        // BossChallengeUI.LoadBoss(index) tier index for Attuned. 0 on the
        // Hornet statue (DISCOVERED.md section 5); re-verify per statue.
        public int TierIndex;
        // GameObject name of a boss-owned projectile worth tracking in the
        // observation (Hornet's thrown needle), or null when the boss has
        // none -- the needle obs fields then always read inactive.
        public string NeedleName;
    }

    public static class BossRegistry
    {
        public static readonly Dictionary<string, BossSpec> All =
            new Dictionary<string, BossSpec>
            {
                ["hornet1"] = new BossSpec
                {
                    Id = "hornet1",
                    Scene = "GG_Hornet_1",
                    ObjectName = "Hornet Boss 1",
                    FsmName = "Control",
                    StatueX = 62.21f,
                    MaxAttunedHp = 1000,
                    TierIndex = 0,
                    NeedleName = "Needle",
                },
                // Measured 2026-08-03, DISCOVERED.md sections 6 and 7.
                // Attuned max HP 650, Ascended 945 -> ceiling 700.
                ["gruz_mother"] = new BossSpec
                {
                    Id = "gruz_mother",
                    Scene = "GG_Gruz_Mother",
                    ObjectName = "Giant Fly",
                    FsmName = "Big Fly Control",
                    // Menu-open readings spanned 27.96-31.15; the macro's
                    // +/-0.5 settle window around 28.0 left the knight at
                    // 27.7-27.9, just outside the interact region's left
                    // edge (statue-menu stalls, 2026-08-03 smoke). 28.6
                    // keeps the whole settle window inside the evidence.
                    StatueX = 28.6f,
                    MaxAttunedHp = 700,
                    TierIndex = 0,
                    NeedleName = null,   // no tracked projectile
                },
                // Measured 2026-08-05, DISCOVERED.md section 8.
                // Attuned max HP 650, Ascended 1000 -> ceiling 700.
                ["gorb"] = new BossSpec
                {
                    Id = "gorb",
                    Scene = "GG_Ghost_Gorb",
                    ObjectName = "Ghost Warrior Slug",
                    FsmName = "Attacking",
                    // Settled menu-open readings 126.22-126.25 (first-approach
                    // outlier 134.66 excluded); the macro's +/-0.5 settle
                    // window around 126.23 (125.73-126.73) contains all of
                    // them (Gruz 28.0->28.6 lesson satisfied).
                    StatueX = 126.23f,
                    MaxAttunedHp = 700,
                    TierIndex = 0,   // re-verify at this statue during smoke
                    // No trackable projectile: "Shot Slug Spear(Clone)" is
                    // logged per-shot (96 instance ids in GG_Ghost_Gorb, not
                    // a stable find-by-name target); the "Spike Collider"
                    // objects are arena hazards, not a boss projectile.
                    NeedleName = null,
                },
            };

        // The boss the current/next episode fights. Defaults to hornet1 so
        // pre-reset code paths (overlay, early reads) always have a spec;
        // set by EpisodeManager whenever a reset message is accepted.
        public static BossSpec Current = All["hornet1"];

        public static bool TrySet(string id)
        {
            if (id != null && All.TryGetValue(id, out var spec))
            {
                Current = spec;
                return true;
            }
            return false;
        }
    }
}
