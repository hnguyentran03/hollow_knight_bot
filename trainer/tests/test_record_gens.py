import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "scripts"))
import record_gens  # noqa: E402  (path insert must precede this import)


def _fake_run(tmp_path, gens):
    (tmp_path / "checkpoints").mkdir()
    manifest = tmp_path / "generations.jsonl"
    lines = []
    for g in gens:
        (tmp_path / "checkpoints" / f"gen_{g:04d}.zip").touch()
        (tmp_path / "checkpoints" / f"gen_{g:04d}_vecnorm.pkl").touch()
        lines.append('{"gen": %d}' % g)
    manifest.write_text("\n".join(lines) + "\n")
    return tmp_path


def test_select_gens_every_takes_stride_plus_last(tmp_path):
    run = _fake_run(tmp_path, [1, 2, 3, 4, 5, 6, 7])
    assert record_gens.select_gens(run, every=3) == [1, 4, 7]
    assert record_gens.select_gens(run, every=2) == [1, 3, 5, 7]


def test_select_gens_always_includes_the_newest(tmp_path):
    run = _fake_run(tmp_path, [1, 2, 3, 4, 5])
    assert record_gens.select_gens(run, every=3)[-1] == 5


def test_select_gens_explicit_list_requires_files(tmp_path):
    run = _fake_run(tmp_path, [10, 20])
    assert record_gens.select_gens(run, gens=[20, 10]) == [10, 20]
    with pytest.raises(FileNotFoundError, match="15"):
        record_gens.select_gens(run, gens=[15])


def test_select_gens_skips_missing_checkpoint_files(tmp_path):
    run = _fake_run(tmp_path, [1, 2, 3])
    (run / "checkpoints" / "gen_0002.zip").unlink()
    assert record_gens.select_gens(run, every=1) == [1, 3]


def test_parser_shape():
    args = record_gens.build_parser().parse_args(
        ["--run-dir", "/x", "--every", "50", "--episodes", "2",
         "--headless"])
    assert args.every == 50 and args.episodes == 2 and args.headless
    args = record_gens.build_parser().parse_args(
        ["--run-dir", "/x", "--gens", "5,10"])
    assert args.gens == [5, 10]
    with pytest.raises(SystemExit):    # --every and --gens are exclusive
        record_gens.build_parser().parse_args(
            ["--run-dir", "/x", "--every", "5", "--gens", "1,2"])
    with pytest.raises(SystemExit):    # one of them is required
        record_gens.build_parser().parse_args(["--run-dir", "/x"])
