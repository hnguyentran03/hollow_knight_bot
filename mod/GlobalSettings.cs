// mod/GlobalSettings.cs
namespace HKRLBot
{
    // Persisted by the Modding API across game restarts
    // (HKRLBotMod : IGlobalSettings<GlobalSettings>).
    public class GlobalSettings
    {
        // The selected export's NAME (its directory under <root>/exports),
        // never a menu index: deleting or reordering exports must not
        // silently shift the selection. "" means nothing selected.
        public string SelectedBot = "";
    }
}
