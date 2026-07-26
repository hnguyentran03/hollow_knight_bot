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
            // Win detection must be event-driven: the Hall of Gods death
            // sequence tears the boss GameObject down faster than
            // EpisodeManager's once-per-hold-window sampling can observe
            // Present && Hp <= 0 (see StateReader.NoteDeath). Recording the
            // death BEFORE orig runs means even a Die() that throws or
            // destroys the object immediately still registers.
            On.HealthManager.Die += (orig, self, attackDirection, attackType, ignoreEvasion) =>
            {
                Reader.NoteDeath(self);
                orig(self, attackDirection, attackType, ignoreEvasion);
            };
            // Unity pauses unfocused applications by default. Parallel training runs
            // several instances at once and only one can hold focus, so without this
            // every background instance freezes and its trainer socket times out.
            Application.runInBackground = true;
            Server.Start();
            Log("HKRLBot initialized");
        }
    }
}
