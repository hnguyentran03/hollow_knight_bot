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
        public VirtualInput Input = new VirtualInput();
        public BridgeServer Server = new BridgeServer();

        public override string GetVersion() => "0.1.0";

        public override void Initialize()
        {
            Instance = this;
            Root = new GameObject("HKRLBot");
            Object.DontDestroyOnLoad(Root);
            Root.AddComponent<OverlayUI>();
            Root.AddComponent<EpisodeManager>();
            UnityEngine.SceneManagement.SceneManager.activeSceneChanged +=
                (_, _2) => Reader.OnSceneChange();
            On.HeroController.Start += (orig, self) => { orig(self); Input.Attach(); };
            // Unity pauses unfocused applications by default. Parallel training runs
            // several instances at once and only one can hold focus, so without this
            // every background instance freezes and its trainer socket times out.
            Application.runInBackground = true;
            Server.Start();
            Log("HKRLBot initialized");
        }
    }
}
