// mod/OverlayUI.cs
using UnityEngine;

namespace HKRLBot
{
    public class OverlayUI : MonoBehaviour
    {
        private bool show = true;

        private bool wiggle;
        private int frame;

        // Highest boss HP observed this fight. The reader only exposes the boss's
        // CURRENT hp (HealthManager.hp via reflection) -- there is no max-HP field
        // anywhere in StateReader/BossState -- so we can't fill the boss bar against
        // a known maximum. Instead we latch the largest value seen: the boss spawns
        // at full health, so the very first read establishes the max and every later
        // read fills against it. Reset to 0 whenever the boss is absent so the next
        // fight re-establishes its own max from full.
        private int bossMaxHp;

        // Chip label for the boss's tracked projectile, uppercased from the
        // registry's object name ("Needle" -> "NEEDLE"). Cached because
        // OnGUI runs several times a frame and ToUpperInvariant allocates.
        private string projectileLabel;
        private string projectileLabelSource;

        // 1x1 white texture reused for every filled rectangle (panel bg, bar tracks,
        // bar fills, chips). Standard IMGUI trick: tint it per-draw with GUI.color and
        // stretch it to any Rect via GUI.DrawTexture -- created ONCE here, never per
        // frame, so OnGUI allocates no textures.
        private Texture2D tex;

        private GUIStyle title;      // panel title
        private GUIStyle header;     // section headers (KNIGHT / BOSS / CONTROLS)
        private GUIStyle body;       // normal rows
        private GUIStyle dim;        // muted labels / units / empty states
        private GUIStyle stateName;  // highlighted boss FSM state name
        private GUIStyle chip;       // centered text inside the boolean-flag chips

        // The game's own legacy Font assets, hunted by name at runtime.
        // IMGUI cannot use TextMeshPro fonts, so only legacy Font assets
        // qualify; whether any exist is measured (census log below), never
        // assumed. Null until found; styles fall back to the IMGUI default.
        private Font serifTitle;   // Trajan-ish: title + section headers
        private Font serifBody;    // Perpetua-ish: body rows, dims, chips
        private bool fontsResolved;
        private bool censusLogged;

        // Precomputed colors -- Color is a struct, so these live in the type's field
        // storage and cost no per-frame GC. Nothing below allocates a Color in a loop.
        private static readonly Color PanelBg   = new Color(0.05f, 0.06f, 0.09f, 0.86f);
        private static readonly Color Accent    = new Color(1.00f, 0.75f, 0.35f, 1.00f);
        private static readonly Color HeaderBg  = new Color(1.00f, 1.00f, 1.00f, 0.06f);
        private static readonly Color TrackCol  = new Color(1.00f, 1.00f, 1.00f, 0.12f);
        private static readonly Color HpGreen   = new Color(0.30f, 0.85f, 0.35f, 0.95f);
        private static readonly Color HpRed     = new Color(0.90f, 0.25f, 0.20f, 0.95f);
        private static readonly Color SoulBlue  = new Color(0.35f, 0.65f, 1.00f, 0.95f);
        private static readonly Color BossHp    = new Color(0.88f, 0.28f, 0.52f, 0.95f);
        private static readonly Color ChipOff   = new Color(1.00f, 1.00f, 1.00f, 0.05f);
        private static readonly Color ChipDim   = new Color(1.00f, 1.00f, 1.00f, 0.30f);
        private static readonly Color GroundOn  = new Color(0.30f, 0.80f, 0.40f, 0.90f);
        private static readonly Color DashOn    = new Color(0.30f, 0.75f, 0.95f, 0.90f);
        private static readonly Color InvulnOn  = new Color(0.95f, 0.80f, 0.25f, 0.90f);
        private static readonly Color FaceCol   = new Color(0.45f, 0.55f, 0.85f, 0.90f);
        private static readonly Color DeadOn    = new Color(0.90f, 0.25f, 0.25f, 0.95f);

        // Layout constants (pixels). Kept fixed so numeric columns don't reflow.
        private const float W        = 344f; // panel width
        private const float Margin   = 12f;  // gap from screen edge
        private const float Pad      = 10f;  // inner panel padding
        private const float TitleH   = 26f;
        private const float HeaderH  = 22f;
        private const float RowH     = 18f;
        private const float BarRowH  = 20f;
        private const float ChipRowH = 20f;
        private const float Gap      = 8f;
        private const float BarH     = 12f;
        private const float ChipH    = 16f;

        // Overall HUD magnification. Applied once via the GUI matrix in OnGUI so
        // this single factor scales the fonts and the whole layout together --
        // bump it to make the entire overlay bigger or smaller without touching
        // any of the pixel constants above.
        private const float Scale    = 1.65f;

        private void Awake()
        {
            tex = new Texture2D(1, 1);
            tex.SetPixel(0, 0, Color.white);
            tex.Apply();
            tex.hideFlags = HideFlags.HideAndDontSave; // don't let it leak into the scene

            title = new GUIStyle { fontSize = 15, fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleLeft };
            title.normal.textColor = new Color(0.96f, 0.92f, 0.82f);

            header = new GUIStyle { fontSize = 12, fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleLeft };
            header.normal.textColor = Accent;

            body = new GUIStyle { fontSize = 12, alignment = TextAnchor.MiddleLeft };
            body.normal.textColor = new Color(0.85f, 0.90f, 0.88f);

            dim = new GUIStyle { fontSize = 12, alignment = TextAnchor.MiddleLeft };
            dim.normal.textColor = new Color(0.60f, 0.66f, 0.68f);

            stateName = new GUIStyle { fontSize = 12, fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleLeft };
            stateName.normal.textColor = Accent;

            chip = new GUIStyle { fontSize = 10, fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleCenter };
            chip.normal.textColor = Color.white;

            TryResolveFonts();
        }

        // One census log line (every loaded legacy Font by name), then pick
        // by case-insensitive substring: "trajan" for titles, "perpetua"
        // for body; if only one family matches it serves both. Fonts load
        // with scenes, so this is retried on every scene change until it
        // succeeds (see OnEnable/OnDisable) -- never per frame.
        private void TryResolveFonts()
        {
            var fonts = Resources.FindObjectsOfTypeAll<Font>();
            if (!censusLogged)
            {
                var names = new System.Text.StringBuilder("OverlayUI font census:");
                foreach (var f in fonts) names.Append(' ').Append(f.name).Append(';');
                HKRLBotMod.Instance.Log(names.ToString());
                censusLogged = true;
            }
            foreach (var f in fonts)
            {
                var n = f.name.ToLowerInvariant();
                if (serifTitle == null && n.Contains("trajan")) serifTitle = f;
                if (serifBody == null && n.Contains("perpetua")) serifBody = f;
            }
            if (serifTitle == null) serifTitle = serifBody;
            if (serifBody == null) serifBody = serifTitle;
            if (serifTitle == null) return; // nothing usable yet; retry on next scene
            fontsResolved = true;
            title.font = serifTitle; header.font = serifTitle;
            body.font = serifBody; dim.font = serifBody;
            stateName.font = serifBody; chip.font = serifBody;
            HKRLBotMod.Instance.Log(
                $"OverlayUI fonts: title={serifTitle.name} body={serifBody.name}");
        }

        private void OnEnable()
        {
            UnityEngine.SceneManagement.SceneManager.activeSceneChanged += OnSceneChanged;
        }

        private void OnDisable()
        {
            UnityEngine.SceneManagement.SceneManager.activeSceneChanged -= OnSceneChanged;
        }

        private void OnSceneChanged(UnityEngine.SceneManagement.Scene from,
                                    UnityEngine.SceneManagement.Scene to)
        {
            if (!fontsResolved) TryResolveFonts();
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.F1)) show = !show;
            if (Input.GetKeyDown(KeyCode.F3)) FSMLogger.LogAll();
            if (Input.GetKeyDown(KeyCode.F4)) DiscoveryLogger.Toggle();

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

        // Tint the shared 1x1 texture and stretch it to fill rect. Saves/restores
        // GUI.color so it never bleeds into the labels drawn afterward.
        private void DrawRect(Rect rect, Color color)
        {
            var prev = GUI.color;
            GUI.color = color;
            GUI.DrawTexture(rect, tex);
            GUI.color = prev;
        }

        // A track with a proportional fill drawn over it.
        private void DrawBar(Rect rect, float frac, Color fill)
        {
            DrawRect(rect, TrackCol);
            frac = Mathf.Clamp01(frac);
            if (frac > 0f)
                DrawRect(new Rect(rect.x, rect.y, rect.width * frac, rect.height), fill);
        }

        // A lit/dim status chip: filled with onColor when active, near-invisible when
        // not, label brightens/dims to match.
        private void DrawChip(Rect rect, string label, bool on, Color onColor)
        {
            DrawRect(rect, on ? onColor : ChipOff);
            chip.normal.textColor = on ? Color.white : ChipDim;
            GUI.Label(rect, label, chip);
        }

        private void OnGUI()
        {
            if (!show) return;

            // Magnify the whole HUD uniformly (fonts + layout) by scaling the GUI
            // matrix. Everything below is drawn in unscaled pixel coordinates that
            // this matrix then blows up by `Scale`, so no per-widget size needs to
            // change. Set every OnGUI so it can never leak into other IMGUI draws.
            GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity,
                                       new Vector3(Scale, Scale, 1f));

            var r = HKRLBotMod.Instance.Reader;
            var k = r.ReadKnight();
            var b = r.ReadBoss();

            // Track boss max HP (see field comment). Reset when the boss is gone.
            if (!b.Present) bossMaxHp = 0;
            else if (b.Hp > bossMaxHp) bossMaxHp = b.Hp;

            // --- Height depends on which states are readable, so measure first, then
            //     draw the background panel behind everything. ---
            float knightH = k == null ? RowH : (BarRowH * 2 + RowH * 2 + ChipRowH);
            float bossH   = !b.Present ? RowH : (BarRowH + RowH * 3 + (BossRegistry.Current.ProjectileName != null ? RowH : 0));
            float total   = Pad + TitleH
                          + Gap + HeaderH + knightH
                          + Gap + HeaderH + bossH
                          + Gap + HeaderH + RowH
                          + Pad;

            // Right-anchored: recomputed every OnGUI so the panel stays glued to
            // the right edge at any resolution. Screen.width is divided by Scale
            // because the coordinates here live in the pre-scale space the GUI
            // matrix magnifies -- the usable width in that space is Screen.width/Scale.
            float x = Screen.width / Scale - W - Margin;
            float top = Margin;
            float left = x + Pad;
            float cw = W - Pad * 2;

            DrawRect(new Rect(x, top, W, total), PanelBg);
            DrawRect(new Rect(x, top, 3f, total), Accent); // left accent stripe

            float cy = top + Pad;
            GUI.Label(new Rect(left, cy, cw, TitleH), "HKRLBOT · DEBUG HUD", title);
            cy += TitleH + Gap;

            // ---------------- KNIGHT ----------------
            cy = SectionHeader(left, cy, cw, "KNIGHT");
            if (k == null)
            {
                GUI.Label(new Rect(left, cy, cw, RowH), "(none)", dim);
                cy += RowH;
            }
            else
            {
                // Knight HP: max is 9 (the RL env normalizes khp/9). Bar lerps
                // red -> green with health so a glance reads the danger level.
                float hpFrac = k.Hp / 9f;
                cy = BarRow(left, cy, cw, "HP", $"{k.Hp}/9", hpFrac, Color.Lerp(HpRed, HpGreen, Mathf.Clamp01(hpFrac)));
                // Soul: PlayerData.MPCharge caps at 99.
                cy = BarRow(left, cy, cw, "SOUL", $"{k.Soul}", k.Soul / 99f, SoulBlue);

                GUI.Label(new Rect(left, cy, cw, RowH), $"pos   x {k.X,8:F2}   y {k.Y,8:F2}", body);
                cy += RowH;
                GUI.Label(new Rect(left, cy, cw, RowH), $"vel   x {k.Vx,8:F2}   y {k.Vy,8:F2}", body);
                cy += RowH;

                // Boolean flags as color-coded chips instead of True/False words.
                float chipW = (cw - 4 * 4) / 5f;
                float chx = left;
                float chy = cy + (ChipRowH - ChipH) / 2f;
                DrawChip(new Rect(chx, chy, chipW, ChipH), "GRND", k.OnGround, GroundOn); chx += chipW + 4;
                DrawChip(new Rect(chx, chy, chipW, ChipH), "DASH", k.Dashing, DashOn);    chx += chipW + 4;
                DrawChip(new Rect(chx, chy, chipW, ChipH), "INV",  k.Invuln, InvulnOn);   chx += chipW + 4;
                // Facing is always "on" (it's a direction, not an alarm); the label
                // shows which way, R with a right arrow or L with a left arrow.
                DrawChip(new Rect(chx, chy, chipW, ChipH), k.FacingRight ? "R »" : "« L", true, FaceCol); chx += chipW + 4;
                DrawChip(new Rect(chx, chy, chipW, ChipH), "DEAD", k.Dead, DeadOn);
                cy += ChipRowH;
            }

            cy += Gap;

            // ---------------- BOSS ----------------
            cy = SectionHeader(left, cy, cw, "BOSS");
            if (!b.Present)
            {
                GUI.Label(new Rect(left, cy, cw, RowH), "(none present)", dim);
                cy += RowH;
            }
            else
            {
                float bFrac = bossMaxHp > 0 ? (float)b.Hp / bossMaxHp : 0f;
                cy = BarRow(left, cy, cw, "HP", $"{b.Hp}/{bossMaxHp}", bFrac, BossHp);

                GUI.Label(new Rect(left, cy, 46f, RowH), "state", dim);
                GUI.Label(new Rect(left + 46f, cy, cw - 46f, RowH),
                          string.IsNullOrEmpty(b.FsmState) ? "?" : b.FsmState, stateName);
                cy += RowH;

                GUI.Label(new Rect(left, cy, cw, RowH), $"pos   x {b.X,8:F2}   y {b.Y,8:F2}", body);
                cy += RowH;
                GUI.Label(new Rect(left, cy, cw, RowH), $"vel   x {b.Vx,8:F2}   y {b.Vy,8:F2}", body);
                cy += RowH;

                // Projectile: only bosses that define a tracked projectile
                // (BossRegistry ProjectileName) get the row at all.
                string projName = BossRegistry.Current.ProjectileName;
                if (projName != null)
                {
                    if (!ReferenceEquals(projName, projectileLabelSource))
                    {
                        projectileLabelSource = projName;
                        projectileLabel = projName.ToUpperInvariant();
                    }
                    float ny = cy + (RowH - ChipH) / 2f;
                    DrawChip(new Rect(left, ny, 68f, ChipH), projectileLabel, b.ProjectileActive, InvulnOn);
                    GUI.Label(new Rect(left + 74f, cy, cw - 74f, RowH),
                              b.ProjectileActive ? $"x {b.ProjectileX,7:F2}   y {b.ProjectileY,7:F2}" : "(inactive)",
                              b.ProjectileActive ? body : dim);
                    cy += RowH;
                }
            }

            cy += Gap;

            // ---------------- CONTROLS ----------------
            cy = SectionHeader(left, cy, cw, "CONTROLS");
            GUI.Label(new Rect(left, cy, cw - 70f, RowH), "F1 show · F2 wiggle · F3 fsm-log · F4 discovery", dim);
            float wy = cy + (RowH - ChipH) / 2f;
            DrawChip(new Rect(left + cw - 66f, wy, 66f, ChipH), wiggle ? "WIGGLE" : "IDLE", wiggle, DashOn);
        }

        // Draws a section header band and returns the y below it.
        private float SectionHeader(float left, float cy, float cw, string label)
        {
            DrawRect(new Rect(left, cy, cw, HeaderH), HeaderBg);
            GUI.Label(new Rect(left + 6f, cy, cw - 6f, HeaderH), label, header);
            return cy + HeaderH;
        }

        // Draws a "LABEL [====bar====] value" meter row and returns the y below it.
        private float BarRow(float left, float cy, float cw, string label, string value, float frac, Color fill)
        {
            GUI.Label(new Rect(left, cy, 42f, BarRowH), label, dim);
            Rect barR = new Rect(left + 44f, cy + (BarRowH - BarH) / 2f, cw - 44f - 52f, BarH);
            DrawBar(barR, frac, fill);
            GUI.Label(new Rect(barR.xMax + 8f, cy, 48f, BarRowH), value, body);
            return cy + BarRowH;
        }
    }
}
