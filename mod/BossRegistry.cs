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
        // Knight's grounded Y standing at this boss's statue in
        // GG_Workshop, measured like StatueX. float.NaN (the default)
        // means "ground-floor statue; keep the legacy walk-only
        // navigation" -- hornet1, gruz_mother, and false_knight are
        // ground-floor and proven working on the walk. The upper-walkway
        // bosses (gorb, marmu, soul_warrior) carry the measured stand
        // Y 36.41 (DISCOVERED.md section 12), which the reset macro
        // teleports to -- the walk cannot climb between floors.
        public float StatueY = float.NaN;
        // Ceiling for backstop B (wrong-difficulty detection): safely above
        // the boss's measured Attuned max HP, below its next tier's.
        public int MaxAttunedHp;
        // BossChallengeUI.LoadBoss(index) tier index for Attuned. 0 on the
        // Hornet statue (DISCOVERED.md section 5); re-verify per statue.
        public int TierIndex;
        // GameObject name of the boss's single tracked projectile in the
        // observation (Hornet's thrown needle), or null when the boss has
        // none -- the projectile obs fields then always read inactive.
        public string ProjectileName;
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
                    ProjectileName = "Needle",
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
                    ProjectileName = null,   // no tracked projectile
                },
                // Measured 2026-08-05, DISCOVERED.md section 8.
                // Attuned max HP 650, Ascended 1000 -> ceiling 700.
                ["gorb"] = new BossSpec
                {
                    Id = "gorb",
                    Scene = "GG_Ghost_Gorb",
                    ObjectName = "Ghost Warrior Slug",
                    FsmName = "Attacking",
                    // smoke-gorb-2 (2026-08-05): macro settled the knight at
                    // 125.90-126.06, just left of the interact region --
                    // the challenge menu never opened. Evidence readings
                    // (126.21-126.25) all came from one standing spot, the
                    // region's left edge -- the same miss Gruz's fix
                    // corrected. 126.8 (left edge + 0.6, the same margin
                    // Gruz's fix used past its lowest reading) is
                    // extrapolated, not measured from a second standing
                    // spot -- the only extrapolated value of the four; the
                    // smoke run verifies it.
                    StatueX = 126.8f,
                    // upper-walkway stand, DISCOVERED.md section 12
                    StatueY = 36.41f,
                    MaxAttunedHp = 700,
                    TierIndex = 0,   // re-verify at this statue during smoke
                    // No trackable projectile: "Shot Slug Spear(Clone)" is
                    // logged per-shot (96 instance ids in GG_Ghost_Gorb, not
                    // a stable find-by-name target); the "Spike Collider"
                    // objects are arena hazards, not a boss projectile.
                    ProjectileName = null,
                },
                // Measured 2026-08-05, DISCOVERED.md section 9.
                // Attuned max HP 750, Ascended 1000 -> ceiling 800.
                ["soul_warrior"] = new BossSpec
                {
                    Id = "soul_warrior",
                    Scene = "GG_Mage_Knight",
                    ObjectName = "Mage Knight",
                    FsmName = "Mage Knight",
                    // smoke-gorb-2 (2026-08-05) diagnosis: the registered
                    // 34.01 was a left-edge-only reading. Two-ended evidence
                    // (34.01, 37.19) gives a midpoint of 35.6; the macro's
                    // +/-0.5 settle window (35.1-36.1) sits fully inside the
                    // evidence span.
                    StatueX = 35.6f,
                    StatueY = 36.41f,
                    MaxAttunedHp = 800,
                    TierIndex = 0,   // re-verify at this statue during smoke
                    ProjectileName = null,   // no persistent projectile; evidence in DISCOVERED.md section 9
                },
                // Measured 2026-08-05, DISCOVERED.md section 10.
                // Attuned max HP 416, Ascended 600 -> ceiling 450.
                ["marmu"] = new BossSpec
                {
                    Id = "marmu",
                    Scene = "GG_Ghost_Marmu",
                    ObjectName = "Ghost Warrior Marmu",
                    FsmName = "Control",
                    // smoke-gorb-2 (2026-08-05) diagnosis: the registered
                    // 91.34 was a left-edge-only reading. Two-ended evidence
                    // (91.34, 94.52) gives a midpoint of 92.9; the macro's
                    // +/-0.5 settle window (92.4-93.4) sits fully inside the
                    // evidence span.
                    StatueX = 92.9f,
                    StatueY = 36.41f,
                    MaxAttunedHp = 450,
                    TierIndex = 0,   // re-verify at this statue during smoke
                    ProjectileName = null,   // no projectile candidates; evidence in DISCOVERED.md section 10
                },
                // Measured 2026-08-05, DISCOVERED.md section 11.
                // Attuned max HP 260, Ascended 560, SAME scene as Attuned
                // (GG_False_Knight) -- unlike every other registered boss,
                // this boss's tiers share one scene, so the scene name
                // cannot distinguish them. MaxAttunedHp (ceiling 300) is
                // therefore the ONLY wrong-tier guard for this boss.
                ["false_knight"] = new BossSpec
                {
                    Id = "false_knight",
                    Scene = "GG_False_Knight",
                    ObjectName = "False Knight New",
                    FsmName = "FalseyControl",
                    // smoke-gorb-2 (2026-08-05) diagnosis: the registered
                    // 52.07 was a left-edge-only reading. Two-ended evidence
                    // (52.07, 55.19) gives a midpoint of 53.6; the macro's
                    // +/-0.5 settle window (53.1-54.1) sits fully inside the
                    // evidence span.
                    StatueX = 53.6f,
                    MaxAttunedHp = 300,
                    TierIndex = 0,   // re-verify at this statue during smoke
                    ProjectileName = null,   // projectiles are per-shot clones; evidence in DISCOVERED.md section 11
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
