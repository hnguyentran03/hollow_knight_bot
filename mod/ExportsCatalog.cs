// mod/ExportsCatalog.cs
using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;

namespace HKRLBot
{
    // Lists <root>/exports/ for the in-game bot selector. The mod reads
    // the disk directly (no protocol involvement) so the menu is
    // populated even before the play daemon starts; the list is a
    // snapshot taken whenever the Mods menu is built.
    public static class ExportsCatalog
    {
        public struct Entry
        {
            public string Name;   // directory name == export name
            public string Label;  // "name · Boss Display" for the menu
        }

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

        public static List<Entry> List()
        {
            var entries = new List<Entry>();
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
                    var name = Path.GetFileName(d);
                    var label = name;
                    try
                    {
                        var manifest = JObject.Parse(
                            File.ReadAllText(Path.Combine(d, "manifest.json")));
                        var display = (string)manifest["boss_display"];
                        if (!string.IsNullOrEmpty(display))
                            label = name + " · " + display;
                    }
                    catch (Exception)
                    {
                        // Unreadable manifest: degrade to the bare name.
                    }
                    entries.Add(new Entry { Name = name, Label = label });
                }
            }
            catch (Exception)
            {
                // Unreadable exports dir (e.g. an IOException off the
                // filesystem): degrade to whatever entries were already
                // collected rather than breaking the Mods menu build.
            }
            entries.Sort((a, b) => string.CompareOrdinal(a.Name, b.Name));
            return entries;
        }
    }
}
