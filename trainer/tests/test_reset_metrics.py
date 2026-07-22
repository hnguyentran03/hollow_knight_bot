import json

import pytest

from hkrl.reset_metrics import (
    append_reset_span, exclusive_freeze_fraction, freeze_fraction,
    read_reset_intervals, read_reset_spans, report_run, reset_log_path,
    resolve_run_params, summarize_freeze,
)


def _write_run(tmp_path, instances=2, wall=600.0):
    (tmp_path / "config.jsonl").write_text(
        json.dumps({"instances": instances, "run_id": "r"}) + "\n")
    (tmp_path / "generations.jsonl").write_text(
        json.dumps({"gen": 1, "wall_clock_s": 100.0}) + "\n" +
        json.dumps({"gen": 2, "wall_clock_s": wall}) + "\n")


def test_freeze_fraction_charges_n_minus_one_over_n_of_each_span():
    # Two instances, two 10s resets, 100s of rollout wall-clock. Each reset
    # freezes the one sibling for its whole span, so the fleet loses
    # (2-1)/2 * (10 + 10) = 10s out of 100s = 0.10.
    assert freeze_fraction([10.0, 10.0], n_instances=2, wallclock_s=100.0) == 0.1


def test_freeze_fraction_scales_with_instance_count():
    # With three instances a reset freezes two siblings, so the same reset
    # budget costs more: (3-1)/3 * 30 / 100 = 0.2.
    assert freeze_fraction([10.0, 10.0, 10.0], n_instances=3,
                           wallclock_s=100.0) == 0.2


def test_freeze_fraction_is_zero_at_single_instance():
    # N=1 has no sibling to freeze -- the whole feature is a no-op there.
    assert freeze_fraction([10.0, 10.0], n_instances=1, wallclock_s=100.0) == 0.0


def test_freeze_fraction_is_zero_with_no_resets():
    assert freeze_fraction([], n_instances=2, wallclock_s=100.0) == 0.0


def test_freeze_fraction_guards_nonpositive_wallclock():
    # No division-by-zero when the wall-clock is missing/zero.
    assert freeze_fraction([10.0], n_instances=2, wallclock_s=0.0) == 0.0


def test_exclusive_freeze_ignores_fully_overlapping_resets():
    # The parallel cold boot: both instances resetting at once freeze nobody
    # who wasn't already resetting -- and startup is unrecoverable anyway.
    assert exclusive_freeze_fraction([(0.0, 10.0), (0.0, 10.0)],
                                     n_instances=2, wallclock_s=100.0) == 0.0


def test_exclusive_freeze_matches_naive_formula_when_disjoint():
    naive = freeze_fraction([10.0, 10.0], n_instances=2, wallclock_s=100.0)
    assert exclusive_freeze_fraction([(0.0, 10.0), (50.0, 60.0)],
                                     n_instances=2, wallclock_s=100.0
                                     ) == pytest.approx(naive)


def test_exclusive_freeze_charges_only_the_non_overlapping_parts():
    # N=2, [0,10] and [5,15]: during the [5,10] overlap both are resetting,
    # so only [0,5] and [10,15] freeze the lone sibling: 10s * 1/2 / 100.
    assert exclusive_freeze_fraction([(0.0, 10.0), (5.0, 15.0)],
                                     n_instances=2, wallclock_s=100.0
                                     ) == pytest.approx(0.05)


def test_exclusive_freeze_integrates_k_of_n_at_three_instances():
    # N=3, [0,10] and [5,15]: [0,5) one resetting -> 2/3, [5,10) two -> 1/3,
    # [10,15) one -> 2/3: (2/3*5 + 1/3*5 + 2/3*5) / 100 = 25/3 / 100.
    assert exclusive_freeze_fraction([(0.0, 10.0), (5.0, 15.0)],
                                     n_instances=3, wallclock_s=100.0
                                     ) == pytest.approx(25.0 / 3.0 / 100.0)


def test_exclusive_freeze_guards_single_instance_and_zero_wallclock():
    assert exclusive_freeze_fraction([(0.0, 10.0)], 1, 100.0) == 0.0
    assert exclusive_freeze_fraction([(0.0, 10.0)], 2, 0.0) == 0.0


def test_read_reset_intervals_reconstructs_start_from_end_and_span(tmp_path):
    append_reset_span(reset_log_path(tmp_path, 9020), span_s=2.5, t=100.0)
    append_reset_span(reset_log_path(tmp_path, 9021), span_s=1.0, t=105.0)
    assert sorted(read_reset_intervals(tmp_path)) == [(97.5, 100.0),
                                                      (104.0, 105.0)]


def test_reset_log_path_is_per_port_under_the_run_dir(tmp_path):
    # One sidecar per worker keyed by its port, mirroring the run dir's
    # per-session monitor_*.csv convention, so concurrent workers never
    # append to the same file.
    p = reset_log_path(tmp_path, port=9021)
    assert p.parent == tmp_path
    assert p.name == "resets_9021.jsonl"


def test_append_and_read_round_trips_spans_across_ports(tmp_path):
    append_reset_span(reset_log_path(tmp_path, 9020), span_s=2.5, t=100.0)
    append_reset_span(reset_log_path(tmp_path, 9020), span_s=3.0, t=110.0)
    append_reset_span(reset_log_path(tmp_path, 9021), span_s=1.0, t=105.0)
    assert sorted(read_reset_spans(tmp_path)) == [1.0, 2.5, 3.0]


def test_read_reset_spans_skips_a_torn_tail_line(tmp_path):
    # The trainer may be appending while the report reads; a half-written
    # final line is a normal condition, skipped rather than raised (same
    # contract as rundata.read_jsonl).
    path = reset_log_path(tmp_path, 9020)
    append_reset_span(path, span_s=4.0, t=100.0)
    with path.open("a") as f:
        f.write('{"span_s": 5.0, "t":')  # torn, no newline
    assert read_reset_spans(tmp_path) == [4.0]


def test_read_reset_spans_is_empty_when_no_sidecars(tmp_path):
    assert read_reset_spans(tmp_path) == []


def test_summarize_freeze_reports_the_gating_numbers():
    s = summarize_freeze([(0.0, 10.0), (20.0, 40.0), (50.0, 80.0)],
                         n_instances=2, wallclock_s=600.0)
    assert s["n_resets"] == 3
    assert s["total_reset_s"] == 60.0
    assert s["mean_reset_s"] == pytest.approx(20.0)
    assert s["max_reset_s"] == 30.0
    assert s["n_instances"] == 2
    assert s["wallclock_s"] == 600.0
    # (2-1)/2 * 60 / 600 = 0.05; disjoint spans, so both numbers agree.
    assert s["freeze_fraction"] == pytest.approx(0.05)
    assert s["exclusive_freeze_fraction"] == pytest.approx(0.05)


def test_summarize_freeze_handles_no_resets():
    s = summarize_freeze([], n_instances=3, wallclock_s=100.0)
    assert s["n_resets"] == 0
    assert s["total_reset_s"] == 0.0
    assert s["mean_reset_s"] == 0.0
    assert s["max_reset_s"] == 0.0
    assert s["freeze_fraction"] == 0.0
    assert s["exclusive_freeze_fraction"] == 0.0


def test_resolve_run_params_reads_instances_and_final_wallclock(tmp_path):
    _write_run(tmp_path, instances=3, wall=720.0)
    n, wall = resolve_run_params(tmp_path)
    assert n == 3
    # Last generation's session wall-clock is the rollout time to divide by.
    assert wall == 720.0


def _write_monitor(tmp_path, t_start, episode_ends, name="monitor_x"):
    # VecMonitor CSV: a #{t_start} header, a r,l,t column line, then one row
    # per episode whose t is seconds since t_start.
    rows = "\n".join(f"-1.0,100,{t}" for t in episode_ends)
    (tmp_path / f"{name}.monitor.csv").write_text(
        f'#{{"t_start": {t_start}, "env_id": "None"}}\nr,l,t\n{rows}\n')


def test_resolve_run_params_derives_wallclock_from_monitor_csv(tmp_path):
    # No generations.jsonl (run interrupted before its first checkpoint): the
    # rollout wall-clock still comes from the monitor CSV -- last episode's
    # absolute end minus the earliest t_start.
    _write_monitor(tmp_path, t_start=1000.0, episode_ends=[100.0, 250.0, 390.0])
    append_reset_span(reset_log_path(tmp_path, 9020), span_s=10.0, t=1.0)
    n, wall = resolve_run_params(tmp_path)
    assert n == 1  # one sidecar, no config
    assert wall == pytest.approx(390.0)


def test_resolve_run_params_spans_wallclock_across_monitor_sessions(tmp_path):
    # A resumed run has two monitor files; wall-clock spans earliest start to
    # latest episode end across both.
    _write_monitor(tmp_path, t_start=1000.0, episode_ends=[100.0], name="monitor_a")
    _write_monitor(tmp_path, t_start=1500.0, episode_ends=[200.0], name="monitor_b")
    _, wall = resolve_run_params(tmp_path)
    # latest end = 1500+200 = 1700; earliest start = 1000 -> 700
    assert wall == pytest.approx(700.0)


def test_resolve_run_params_falls_back_to_sidecar_count_for_instances(tmp_path):
    # No config.jsonl: infer N from the number of per-port sidecars.
    append_reset_span(reset_log_path(tmp_path, 9020), span_s=1.0, t=1.0)
    append_reset_span(reset_log_path(tmp_path, 9021), span_s=1.0, t=1.0)
    n, wall = resolve_run_params(tmp_path)
    assert n == 2
    assert wall == 0.0  # no manifest -> unknown wall-clock


def test_report_run_ties_spans_params_and_summary_together(tmp_path):
    _write_run(tmp_path, instances=2, wall=600.0)
    append_reset_span(reset_log_path(tmp_path, 9020), span_s=30.0, t=100.0)
    append_reset_span(reset_log_path(tmp_path, 9021), span_s=30.0, t=200.0)
    s = report_run(tmp_path)
    assert s["n_resets"] == 2
    assert s["n_instances"] == 2
    assert s["wallclock_s"] == 600.0
    # (2-1)/2 * 60 / 600 = 0.05, disjoint so exclusive agrees
    assert s["freeze_fraction"] == pytest.approx(0.05)
    assert s["exclusive_freeze_fraction"] == pytest.approx(0.05)


def test_report_run_accepts_overrides(tmp_path):
    append_reset_span(reset_log_path(tmp_path, 9020), span_s=30.0, t=1.0)
    s = report_run(tmp_path, n_instances=4, wallclock_s=300.0)
    assert s["n_instances"] == 4
    assert s["wallclock_s"] == 300.0
    # (4-1)/4 * 30 / 300 = 0.075
    assert s["freeze_fraction"] == pytest.approx(0.075)
