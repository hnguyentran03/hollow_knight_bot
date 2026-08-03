import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from parse_discovery import report, summarize

SAMPLE = """\
[INFO]:[HKRLBot] - DISCOVERY logging ON
[INFO]:[HKRLBot] - DISCOVERY candidate go='Giant Fly' hp=660 scene=GG_Gruz_Mother
[INFO]:[HKRLBot] - DISCOVERY state go='Giant Fly' state='Init'
[INFO]:[HKRLBot] - DISCOVERY state go='Giant Fly' state='Fly'
[INFO]:[HKRLBot] - DISCOVERY state go='Giant Fly' state='Charge Antic'
[INFO]:[HKRLBot] - DISCOVERY state go='Giant Fly' state='Fly'
[INFO]:[HKRLBot] - DISCOVERY arena scene=GG_Gruz_Mother kxRange=[10.00, 20.00] floorY=5.50 maxKy=12.50
[INFO]:[HKRLBot] - DISCOVERY arena scene=GG_Gruz_Mother kxRange=[8.00, 30.00] floorY=5.50 maxKy=14.50
[INFO]:[HKRLBot] - DISCOVERY statue knightX=55.30 scene=GG_Workshop
""".splitlines()


def test_summarize_dedupes_states_preserving_first_seen_order():
    s = summarize(SAMPLE)
    assert s["states"]["Giant Fly"] == ["Init", "Fly", "Charge Antic"]


def test_summarize_tracks_peak_hp_and_last_arena_and_statue():
    s = summarize(SAMPLE)
    assert s["candidates"]["Giant Fly"]["hp"] == 660
    a = s["arenas"]["GG_Gruz_Mother"]
    assert (a["min"], a["max"], a["floor"], a["top"]) == (8.0, 30.0, 5.5, 14.5)
    assert s["statue_xs"] == [55.3]


def test_report_derives_registry_values():
    out = report(summarize(SAMPLE))
    assert "center_x=19.00" in out      # (8+30)/2
    assert "half_w=11.00" in out        # (30-8)/2
    assert "height=9.00" in out         # 14.5-5.5
    assert "go='Giant Fly' peak hp=660" in out
