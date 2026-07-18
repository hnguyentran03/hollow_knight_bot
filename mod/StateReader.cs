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

        public void OnSceneChange()
        {
            bossGo = null; bossHm = null; bossFsm = null; bossRb = null; needleGo = null;
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
                Vx = rb != null ? rb.linearVelocity.x : 0f,
                Vy = rb != null ? rb.linearVelocity.y : 0f,
                Hp = pd.health, Soul = pd.MPCharge,
                OnGround = hc.cState.onGround, Dashing = hc.cState.dashing,
                Invuln = hc.cState.invulnerable, FacingRight = hc.cState.facingRight,
                Dead = hc.cState.dead
            };
        }

        public BossState ReadBoss()
        {
            if (bossGo == null)
            {
                bossGo = GameObject.Find("Hornet Boss 1");
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
                Vx = bossRb != null ? bossRb.linearVelocity.x : 0f,
                Vy = bossRb != null ? bossRb.linearVelocity.y : 0f,
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
