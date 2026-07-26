// mod/StateReader.cs
using System.Reflection;
using Modding;
using UnityEngine;
using UnityEngine.EventSystems;

namespace HKRLBot
{
    public class KnightState
    {
        public float X, Y, Vx, Vy;
        public int Hp, Soul;
        public bool OnGround, Dashing, Invuln, FacingRight, Dead;
    }

    public class BossState
    {
        public bool Present;
        // The boss's HealthManager ran Die() this scene. Event-driven (see
        // StateReader.NoteDeath), so it stays true after the death sequence
        // tears the GameObject down and Present/Hp can no longer say
        // anything -- the signal EpisodeManager's win detection needs.
        public bool Died;
        public float X, Y, Vx, Vy;
        public int Hp;
        public string FsmState = "";
        public float NeedleX, NeedleY;
        public bool NeedleActive;
    }

    public class StateReader
    {
        private GameObject bossGo;
        private HealthManager bossHm;
        private PlayMakerFSM bossFsm;
        private Rigidbody2D bossRb;
        private GameObject needleGo;
        // ReadBoss() is a hot path: EpisodeManager calls it every decision step.
        // GameObject.Find is a full scene-hierarchy scan, so once the boss has
        // been found in the current scene there is no need to re-scan on every
        // call -- this flag latches to skip the scan for the rest of the scene.
        //
        // This must latch ONLY on a successful find, not on every attempt. If it
        // latched unconditionally, a single miss (e.g. the boss GameObject not
        // active yet on the very first frame after activeSceneChanged, due to
        // scene-load ordering or a Godhome entry sequence) would permanently
        // disable boss detection for the rest of the scene -- EpisodeManager.
        // TickReset's fightLive gate could then never become true, so a reset
        // would never complete. So while the boss is genuinely absent, ReadBoss()
        // re-runs GameObject.Find on every call (a few wasted scans during the
        // absent window is cheap and correct); the cache only kicks in once the
        // boss is actually found, which is the steady-state case this
        // optimization exists for. Cleared by OnSceneChange() alongside the
        // other cached handles so a scene change always gets a fresh search.
        private bool bossSearchDone;
        // Set by NoteDeath the moment the boss's HealthManager runs Die().
        // Cleared on scene change: a Godhome retry reloads the arena scene,
        // so scene lifetime IS fight lifetime for this flag.
        private bool bossDied;

        public void OnSceneChange()
        {
            bossGo = null; bossHm = null; bossFsm = null; bossRb = null; needleGo = null;
            bossSearchDone = false;
            bossDied = false;
        }

        // Called from HKRLBotMod's On.HealthManager.Die hook for EVERY
        // HealthManager death in the game (minions, spawned hazards, other
        // scenes); the identity check against the cached bossHm is what
        // narrows it to the boss. Event-driven because polling cannot work
        // here: in Hall of Gods the fatal blow's death sequence tears the
        // boss GameObject down within one ~67ms action-hold window, so no
        // sampled frame ever shows Present && Hp <= 0 -- measured across
        // entire runs (0 won=True in any ModLog against 100%-damage
        // episodes) before this hook existed.
        public void NoteDeath(HealthManager hm)
        {
            if (hm != null && bossHm != null && hm == bossHm) bossDied = true;
        }

        public KnightState ReadKnight()
        {
            var hc = HeroController.instance;
            if (hc == null) return null;
            var pd = PlayerData.instance;
            var rb = hc.GetComponent<Rigidbody2D>();
            var p = hc.transform.position;
            return new KnightState
            {
                X = p.x, Y = p.y,
                // Rigidbody2D.velocity, not .linearVelocity: a Steam update to
                // Hollow Knight replaced the game's Physics2D module and
                // removed linearVelocity entirely. Switching this back to
                // linearVelocity (e.g. to silence a deprecation warning on a
                // newer Unity) breaks the build against the installed game.
                Vx = rb != null ? rb.velocity.x : 0f,
                Vy = rb != null ? rb.velocity.y : 0f,
                Hp = pd.health, Soul = pd.MPCharge,
                OnGround = hc.cState.onGround, Dashing = hc.cState.dashing,
                Invuln = hc.cState.invulnerable, FacingRight = hc.cState.facingRight,
                Dead = hc.cState.dead
            };
        }

        public BossState ReadBoss()
        {
            if (bossGo == null && !bossSearchDone)
            {
                bossGo = GameObject.Find("Hornet Boss 1");
                if (bossGo != null)
                {
                    bossSearchDone = true;
                    bossHm = bossGo.GetComponent<HealthManager>();
                    bossFsm = FSMUtility.LocateFSM(bossGo, "Control");
                    bossRb = bossGo.GetComponent<Rigidbody2D>();
                }
                // else: leave bossSearchDone false so the next ReadBoss() call
                // retries the search instead of latching a false negative.
            }
            if (bossGo == null) return new BossState { Present = false, Died = bossDied };

            if (needleGo == null) needleGo = GameObject.Find("Needle");
            var bp = bossGo.transform.position;
            var s = new BossState
            {
                Present = true,
                Died = bossDied,
                X = bp.x, Y = bp.y,
                // .velocity, not .linearVelocity -- see the note in ReadKnight above.
                Vx = bossRb != null ? bossRb.velocity.x : 0f,
                Vy = bossRb != null ? bossRb.velocity.y : 0f,
                Hp = bossHm != null ? ReflectionHelper.GetField<HealthManager, int>(bossHm, "hp") : 0,
                FsmState = bossFsm != null ? bossFsm.ActiveStateName : ""
            };
            if (needleGo != null && needleGo.activeInHierarchy)
            {
                s.NeedleActive = true;
                s.NeedleX = needleGo.transform.position.x;
                s.NeedleY = needleGo.transform.position.y;
            }
            return s;
        }

        // Whether the boss statue's challenge/difficulty menu is currently
        // open, independent of whether a tier is selected within it. Exposed
        // separately from ReadSelectedChallengeTier() below because a
        // trainer-driven run (2026-07-21) showed those two questions are NOT
        // interchangeable: with the game window unfocused/automated, the
        // menu opens with NO initial EventSystem selection at all (the
        // engine's select-on-open apparently depends on window focus), so
        // ReadSelectedChallengeTier() read -1 through an ENTIRE statue-menu
        // phase in that run -- the same -1 it returns when the menu isn't
        // open yet. A caller that only sees that reader cannot tell "menu
        // not open" apart from "menu open, nothing selected", and the old
        // gate collapsed both into "proceed unchanged", which meant the
        // tier gate never actually verified anything in that run. This
        // method lets EpisodeManager split the two cases: menu open,
        // whatever ReadSelectedChallengeTier() says. Exception-safe: any
        // FindObjectOfType failure reads as "not open" (false), never
        // throws.
        public bool IsChallengeMenuOpen()
        {
            try { return UnityEngine.Object.FindObjectOfType<BossChallengeUI>() != null; }
            catch { return false; }
        }

        // The challenge menu's currently-selected difficulty tier
        // (0=Attuned, 1=Ascended, 2=Radiant), or -1 when the menu is not
        // open or the signal cannot be read. The menu-open check
        // (FindObjectOfType<BossChallengeUI>() != null) is what makes -1
        // trustworthy: a PlayerData field (bossStatueTargetLevel) is
        // readable at all times and would return a stale tier with no menu
        // on screen -- confirmed dead pre-confirm in-game (it read a global
        // -1 through a full tier-cycling session). Called only from the
        // statue-menu macro cycles (a few seconds per reset), so the
        // FindObjectOfType scan cost is bounded. See DISCOVERED.md for how
        // this signal -- EventSystem.current.currentSelectedGameObject
        // compared against the three tier buttons -- was identified
        // in-game (verified 0->1->2->0 while a human cycled tiers).
        //
        // Trainer-context caveat (2026-07-21, verified in a real N=1 run):
        // with the game window unfocused/automated, opening the menu does
        // NOT auto-select a tier the way it does for a focused human
        // session -- currentSelectedGameObject stays null right through the
        // menu-open window, so this reads -1 for the ENTIRE visit, not just
        // before the menu opens. That -1 is still correct (nothing IS
        // selected), but it means -1 no longer distinguishes "menu not open"
        // from "menu open, unselected" in this environment -- callers that
        // need that distinction should check IsChallengeMenuOpen() above
        // first. EpisodeManager's statue-menu gate does exactly that: on an
        // open-but-unselected menu it calls SelectAttunedChallengeTier()
        // itself rather than waiting for a selection that a headless/
        // unfocused game will never produce on its own.
        public int ReadSelectedChallengeTier()
        {
            var ui = UnityEngine.Object.FindObjectOfType<BossChallengeUI>();
            if (ui == null) return -1;   // menu not open
            try
            {
                var cur = EventSystem.current != null ? EventSystem.current.currentSelectedGameObject : null;
                if (cur == null) return -1;
                if (cur == ui.tier1Button.button.gameObject) return 0;
                if (cur == ui.tier2Button.button.gameObject) return 1;
                if (cur == ui.tier3Button.button.gameObject) return 2;
                return -1;
            }
            catch { return -1; }
        }

        // Corrects the challenge menu's selection to Attuned (tier1) by
        // writing the EventSystem's selected GameObject directly. The read
        // side above proved the EventSystem selection IS the highlight (it
        // tracks the highlight live, not just on a selection event, unlike
        // BossChallengeUI's rejected static lastSelectedButton), so setting
        // it selects Attuned with no synthetic navigation input needed.
        // Returns true on success, false on "menu not open" or any
        // exception, so a caller can treat both the same way. Kept here
        // rather than in EpisodeManager to keep all scene/EventSystem
        // access for this signal in one file. See DISCOVERED.md.
        public bool SelectAttunedChallengeTier()
        {
            var ui = UnityEngine.Object.FindObjectOfType<BossChallengeUI>();
            if (ui == null) return false;
            try
            {
                EventSystem.current.SetSelectedGameObject(ui.tier1Button.button.gameObject);
                return true;
            }
            catch { return false; }
        }

        // Requests the Attuned challenge tier BY VALUE, instead of by
        // correcting/confirming a highlight: invokes BossChallengeUI's own
        // `LoadBoss(int level)` (level 0 = Attuned) -- the same method the
        // tier buttons themselves call on click -- via reflection.
        //
        // Why reflection: LoadBoss's member visibility on the disassembled
        // BossChallengeUI is unknown/unverified (could be private, could be
        // an overload set -- there is also a `LoadBoss(int, bool
        // doHideAnim)`), and reflection with an explicit parameter-type
        // filter (new[] { typeof(int) }) sidesteps both problems: it finds
        // the single-int overload regardless of its visibility, and
        // BindingFlags.Public | BindingFlags.NonPublic means we don't have
        // to know or trust which one it is.
        //
        // Why this beats SelectAttunedChallengeTier()+confirm: that path
        // requests Attuned by correcting/confirming an EventSystem highlight,
        // which is exactly the signal a real trainer run (2026-07-21, game
        // unfocused) verified the engine refuses to move --
        // SetSelectedGameObject did not stick across 2 attempts (see
        // DISCOVERED.md). LoadBoss(0) instead asks the menu to load the tier
        // directly, with no EventSystem/focus dependency at all -- it is the
        // same call the tier button's OnClick makes, just invoked ourselves.
        //
        // Exception-safe like this file's other helpers: null UI or any
        // reflection/invoke failure (missing method, target exception from
        // inside LoadBoss, etc.) returns false rather than throwing, so the
        // caller can fall back to the blind confirm.
        public bool ConfirmAttunedChallenge()
        {
            var ui = UnityEngine.Object.FindObjectOfType<BossChallengeUI>();
            if (ui == null) return false;
            try
            {
                var method = ui.GetType().GetMethod(
                    "LoadBoss",
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                    null, new[] { typeof(int) }, null);
                if (method == null) return false;
                method.Invoke(ui, new object[] { 0 });
                return true;
            }
            catch { return false; }
        }
    }
}
