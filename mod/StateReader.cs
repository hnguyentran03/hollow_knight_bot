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
        // Task 6: ReadBoss() became a hot path once EpisodeManager calls it every
        // decision step (previously only OverlayUI called it, once per rendered
        // frame). GameObject.Find is a full scene-hierarchy scan; while the boss is
        // genuinely absent from the current scene (any non-fight scene, or a fight
        // scene before the boss has spawned), bossGo stays null forever and every
        // single ReadBoss() call was re-running that scan. This flag remembers "we
        // already looked for the boss in this scene" so absence is a cheap early-out
        // after the first miss. Cleared by OnSceneChange() alongside the other cached
        // handles so a scene change always gets a fresh search.
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
                bossSearchDone = true;
                if (bossGo != null)
                {
                    bossHm = bossGo.GetComponent<HealthManager>();
                    bossFsm = FSMUtility.LocateFSM(bossGo, "Control");
                    bossRb = bossGo.GetComponent<Rigidbody2D>();
                }
            }
            if (bossGo == null) return new BossState { Present = false };

            if (needleGo == null) needleGo = GameObject.Find("Needle");
            var bp = bossGo.transform.position;
            var s = new BossState
            {
                Present = true,
                X = bp.x, Y = bp.y,
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
