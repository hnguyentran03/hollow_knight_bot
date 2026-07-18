// mod/OverlayUI.cs
using UnityEngine;

namespace HKRLBot
{
    public class OverlayUI : MonoBehaviour
    {
        private bool show = true;
        private static readonly GUIStyle Style = new GUIStyle();

        private bool wiggle;
        private int frame;

        private void Awake()
        {
            Style.fontSize = 18;
            Style.normal.textColor = Color.green;
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.F1)) show = !show;
            if (Input.GetKeyDown(KeyCode.F3)) FSMLogger.LogAll();

            if (Input.GetKeyDown(KeyCode.F2)) { wiggle = !wiggle; if (!wiggle) HKRLBotMod.Instance.Input.Clear(); }
            if (wiggle)
            {
                frame++;
                var b = new ActionButtons();
                b.Left = (frame / 30) % 2 == 0;
                b.Right = !b.Left;
                b.Jump = (frame % 60) < 10;
                b.Attack = (frame % 45) < 3;
                HKRLBotMod.Instance.Input.Apply(b);
            }
        }

        private void OnGUI()
        {
            if (!show) return;
            var r = HKRLBotMod.Instance.Reader;
            var k = r.ReadKnight();
            var b = r.ReadBoss();
            string text = "HKRLBot echo\n";
            text += k == null ? "Knight: (none)\n"
                : $"Knight: x={k.X:F2} y={k.Y:F2} vx={k.Vx:F2} vy={k.Vy:F2} hp={k.Hp} soul={k.Soul}\n" +
                  $"  ground={k.OnGround} dash={k.Dashing} invuln={k.Invuln} right={k.FacingRight} dead={k.Dead}\n";
            text += !b.Present ? "Boss: (none)\n"
                : $"Boss: x={b.X:F2} y={b.Y:F2} hp={b.Hp} state={b.FsmState}\n" +
                  $"  needle active={b.NeedleActive} x={b.NeedleX:F2} y={b.NeedleY:F2}\n";
            GUI.Label(new Rect(10, 10, 900, 200), text, Style);
        }
    }
}
