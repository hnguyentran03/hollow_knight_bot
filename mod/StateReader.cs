// mod/StateReader.cs
using Modding;
using UnityEngine;

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

        public void OnSceneChange()
        {
            bossGo = null; bossHm = null; bossFsm = null; bossRb = null; needleGo = null;
            bossSearchDone = false;
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
            if (bossGo == null) return new BossState { Present = false };

            if (needleGo == null) needleGo = GameObject.Find("Needle");
            var bp = bossGo.transform.position;
            var s = new BossState
            {
                Present = true,
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
    }
}
