// mod/ExportsCatalog.cs
using System;
using System.Collections.Generic;
using System.IO;

namespace HKRLBot
{
    // Lists <root>/exports/ for the in-game bot selector. The mod reads
    // the disk directly (no protocol involvement) so the menu is
    // populated even before the play daemon starts; the list is a
    // snapshot taken whenever the Mods menu is built.
    public static class ExportsCatalog
    {
        // Matches the trainer's --root default (~/hkrl); HKRL_ROOT
        // overrides it, mirroring how HKRL_PORT already reaches the mod.
        public static string Root()
        {
            var env = Environment.GetEnvironmentVariable("HKRL_ROOT");
            if (!string.IsNullOrEmpty(env)) return env;
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "hkrl");
        }

        // Export names, sorted. Bare names only: the Mods menu renders a
        // horizontal option's value right-aligned across the same rect as
        // its label, so anything longer (e.g. a "name · Boss" suffix)
        // collides with the label text.
        public static List<string> List()
        {
            var entries = new List<string>();
            string dir;
            try { dir = Path.Combine(Root(), "exports"); }
            catch (Exception) { return entries; }
            if (!Directory.Exists(dir)) return entries;
            try
            {
                foreach (var d in Directory.GetDirectories(dir))
                {
                    // Only directories that actually hold a model qualify --
                    // a half-deleted export must not be selectable.
                    if (!File.Exists(Path.Combine(d, "model.zip"))) continue;
                    entries.Add(Path.GetFileName(d));
                }
            }
            catch (Exception)
            {
                // Unreadable exports dir (e.g. an IOException off the
                // filesystem): degrade to whatever entries were already
                // collected rather than breaking the Mods menu build.
            }
            entries.Sort(string.CompareOrdinal);
            return entries;
        }
    }
}
