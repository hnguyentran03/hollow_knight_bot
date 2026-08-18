// mod/HKRLBotMod.cs
using System.Collections.Generic;
using Modding;
using UnityEngine;

namespace HKRLBot
{
    public class HKRLBotMod : Mod, IGlobalSettings<GlobalSettings>, IMenuMod
    {
        internal static HKRLBotMod Instance;
        internal GameObject Root;
        public StateReader Reader = new StateReader();
        public VirtualInput Input = new VirtualInput();
        public BridgeServer Server = new BridgeServer();

        // Selection persisted by the Modding API's global-settings
        // lifecycle; read by OverlayUI's F9 handler and HUD.
        internal static GlobalSettings GS = new GlobalSettings();
        public void OnLoadGlobal(GlobalSettings s) { if (s != null) GS = s; }
        public GlobalSettings OnSaveGlobal() => GS;

        public bool ToggleButtonInsideMenu => false;

        public List<IMenuMod.MenuEntry> GetMenuData(IMenuMod.MenuEntry? toggleButtonEntry)
        {
            // Snapshot at menu build (game boot, or a save closed back to
            // the main menu). Exports created mid-session appear on the
            // next build -- accepted trade-off, see the 2026-08-15 spec.
            var exports = ExportsCatalog.List();
            var values = new string[exports.Count == 0 ? 1 : exports.Count];
            if (exports.Count == 0) values[0] = "(no exports)";
            for (int i = 0; i < exports.Count; i++) values[i] = exports[i];
            return new List<IMenuMod.MenuEntry>
            {
                new IMenuMod.MenuEntry(
                    "Bot",
                    values,
                    "Exported bot F9 plays (from " + ExportsCatalog.Root() + "/exports)",
                    i => { if (exports.Count > 0) GS.SelectedBot = exports[i]; },
                    () =>
                    {
                        // Saved-name -> index; a vanished name displays
                        // entry 0 WITHOUT overwriting the setting (only a
                        // deliberate save does).
                        for (int i = 0; i < exports.Count; i++)
                            if (exports[i] == GS.SelectedBot) return i;
                        return 0;
                    })
            };
        }

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
