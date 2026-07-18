// mod/HKRLBotMod.cs
using Modding;
using UnityEngine;

namespace HKRLBot
{
    public class HKRLBotMod : Mod
    {
        internal static HKRLBotMod Instance;
        internal GameObject Root;
        public StateReader Reader = new StateReader();

        public override string GetVersion() => "0.1.0";

        public override void Initialize()
        {
            Instance = this;
            Root = new GameObject("HKRLBot");
            Object.DontDestroyOnLoad(Root);
            Root.AddComponent<OverlayUI>();
            UnityEngine.SceneManagement.SceneManager.activeSceneChanged +=
                (_, _2) => Reader.OnSceneChange();
            Log("HKRLBot initialized");
        }
    }
}
