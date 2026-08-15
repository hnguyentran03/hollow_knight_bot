import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import export_gen  # noqa: E402  (path insert must precede this import)


def _run(tmp_path):
    run = tmp_path / "runs" / "r1"
    ckpt = run / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "gen_0001.zip").write_bytes(b"w")
    (ckpt / "gen_0001_vecnorm.pkl").write_bytes(b"v")
    (run / "generations.jsonl").write_text(
        json.dumps({"gen": 1, "timestep": 15000}) + "\n")
    return run


def test_parser_defaults():
    args = export_gen.build_parser().parse_args(["--run-dir", "/x"])
    assert str(args.run_dir) == "/x"
    assert args.gen is None and args.name is None and args.force is False
    assert args.root == pathlib.Path("~/hkrl").expanduser()


def test_main_exports_and_prints_the_destination(tmp_path, capsys,
                                                 monkeypatch):
    run = _run(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "export_gen.py", "--run-dir", str(run), "--root", str(tmp_path)])
    export_gen.main()
    out = capsys.readouterr().out
    assert "r1_gen0001" in out
    assert (tmp_path / "exports" / "r1_gen0001" / "model.zip").exists()


def test_main_exits_with_the_error_message_on_collision(tmp_path,
                                                        monkeypatch):
    run = _run(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "export_gen.py", "--run-dir", str(run), "--root", str(tmp_path)])
    export_gen.main()
    with pytest.raises(SystemExit) as err:
        export_gen.main()
    assert "already exists" in str(err.value.code)
