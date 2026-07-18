// mod/FSMLogger.cs
using UnityEngine;

namespace HKRLBot
{
    public static class FSMLogger
    {
        // Dumps every live PlayMakerFSM: gameobject, fsm name, active state.
        // Trigger with F3 (wired in OverlayUI) right after dying to find the
        // retry prompt's FSM and event names.
        public static void LogAll()
        {
            foreach (var fsm in Object.FindObjectsOfType<PlayMakerFSM>())
            {
                HKRLBotMod.Instance.Log(
                    $"FSM go='{fsm.gameObject.name}' fsm='{fsm.FsmName}' state='{fsm.ActiveStateName}'");
            }
        }
    }
}
