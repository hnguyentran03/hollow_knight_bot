import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from parse_discovery import report, summarize

SAMPLE = """\
[INFO]:[HKRLBot] - DISCOVERY logging ON
[INFO]:[HKRLBot] - DISCOVERY candidate go='Giant Fly' hp=660 scene=GG_Gruz_Mother
[INFO]:[HKRLBot] - DISCOVERY state go='Giant Fly' fsm='Mega Fly' state='Init'
[INFO]:[HKRLBot] - DISCOVERY state go='Giant Fly' fsm='Mega Fly' state='Fly'
[INFO]:[HKRLBot] - DISCOVERY state go='Giant Fly' fsm='Constrain X' state='Active'
[INFO]:[HKRLBot] - DISCOVERY state go='Giant Fly' fsm='Mega Fly' state='Charge Antic'
[INFO]:[HKRLBot] - DISCOVERY state go='Giant Fly' fsm='Mega Fly' state='Fly'
[INFO]:[HKRLBot] - DISCOVERY arena scene=GG_Gruz_Mother kxRange=[10.00, 20.00] floorY=5.50 maxKy=12.50
[INFO]:[HKRLBot] - DISCOVERY arena scene=GG_Gruz_Mother kxRange=[8.00, 30.00] floorY=5.50 maxKy=14.50
[INFO]:[HKRLBot] - DISCOVERY projectile go='Needle' id=1234 scene=GG_Hornet_1
[INFO]:[HKRLBot] - DISCOVERY projectile go='Shot Gruz' id=-500 scene=GG_Gruz_Mother
[INFO]:[HKRLBot] - DISCOVERY projectile go='Shot Gruz' id=-501 scene=GG_Gruz_Mother
[INFO]:[HKRLBot] - DISCOVERY statue knightX=55.30 scene=GG_Workshop
""".splitlines()


def test_summarize_dedupes_states_per_fsm_preserving_first_seen_order():
    s = summarize(SAMPLE)
    assert s["states"]["Giant Fly", "Mega Fly"] == ["Init", "Fly", "Charge Antic"]
    assert s["states"]["Giant Fly", "Constrain X"] == ["Active"]


def test_summarize_tracks_peak_hp_and_last_arena_and_statue():
    s = summarize(SAMPLE)
    assert s["candidates"]["Giant Fly", "GG_Gruz_Mother"] == 660
    a = s["arenas"]["GG_Gruz_Mother"]
    assert (a["min"], a["max"], a["floor"], a["top"]) == (8.0, 30.0, 5.5, 14.5)
    assert s["statue_xs"] == [55.3]


def test_report_derives_registry_values():
    out = report(summarize(SAMPLE))
    assert "center_x=19.00" in out      # (8+30)/2
    assert "half_w=11.00" in out        # (30-8)/2
    assert "height=9.00" in out         # 14.5-5.5
    assert "go='Giant Fly' scene=GG_Gruz_Mother peak hp=660" in out


def test_summarize_collects_projectile_instance_ids_per_name_and_scene():
    s = summarize(SAMPLE)
    assert s["projectiles"]["Needle", "GG_Hornet_1"] == {1234}
    assert s["projectiles"]["Shot Gruz", "GG_Gruz_Mother"] == {-500, -501}


def test_report_lists_projectile_candidates_with_instance_counts():
    out = report(summarize(SAMPLE))
    assert "go='Needle' scene=GG_Hornet_1 instances=1" in out
    assert "go='Shot Gruz' scene=GG_Gruz_Mother instances=2" in out
