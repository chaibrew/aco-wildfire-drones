import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
from scipy import stats as sps

from experiment.stats import change_interaction, holm_adjust, load_rows, paired_summary

FRACTIONS = (0.1, 0.4, 0.6)
LOCALITIES = ("concentrated", "distributed")
REF, SEL = "aco_full_reset", "aco_selective_reset"


def make_rows(trials=6, scenarios=("synthetic",), gain_conc=0.0, gain_dist=0.0, jitter=True):
    rows = []
    for si, scenario in enumerate(scenarios):
        for trial in range(trials):
            for f in FRACTIONS:
                for loc in LOCALITIES:
                    ref = 2.0 + 0.1 * trial + 0.5 * si + 0.05 * f
                    noise = 0.03 * (((trial * 5 + int(f * 10)) % 4) - 1.5) if jitter else 0.0
                    gain = gain_conc if loc == "concentrated" else gain_dist
                    base = dict(phase="final", scenario=scenario, trial=trial, target_fraction=f, locality=loc)
                    rows.append({**base, "planner": REF, "margin": ref})
                    rows.append({**base, "planner": SEL, "margin": ref - gain + noise})
    return rows


def sel_row(rows_out):
    return next(r for r in rows_out if r["planner"] == SEL)


def test_pooled_grouping_tests_one_value_per_trial_not_per_row():
    rows = make_rows(gain_conc=0.2, gain_dist=0.1)
    summary = paired_summary(rows, "margin", REF, "locality")
    out = sel_row([r for r in summary if r["locality"] == "concentrated"])
    assert out["n_pairs"] == 6 * len(FRACTIONS)
    assert out["n_clusters"] == 6
    diffs = []
    for trial in range(6):
        d = [a["margin"] - b["margin"] for a in rows for b in rows
             if a["planner"] == SEL and b["planner"] == REF and a["trial"] == b["trial"] == trial
             and a["target_fraction"] == b["target_fraction"] and a["locality"] == b["locality"] == "concentrated"]
        diffs.append(statistics.mean(d))
    assert out["mean_diff"] == pytest.approx(statistics.mean(diffs))
    assert out["t_p"] == pytest.approx(float(sps.ttest_1samp(diffs, 0.0).pvalue))


def test_a_constant_scenario_label_does_not_collapse_the_clusters():
    rows = make_rows(trials=5)
    assert {r["scenario"] for r in rows} == {"synthetic"}
    for grouping in ("all", "locality", "fraction"):
        assert sel_row(paired_summary(rows, "margin", REF, grouping))["n_clusters"] == 5


def test_cell_grouping_keeps_one_observation_per_trial():
    rows = make_rows(trials=6, gain_conc=0.2)
    out = [r for r in paired_summary(rows, "margin", REF, "cell") if r["planner"] == SEL]
    assert len(out) == len(FRACTIONS) * len(LOCALITIES)
    assert all(r["n_pairs"] == r["n_clusters"] == 6 for r in out)


def test_same_trial_number_in_different_scenarios_are_different_clusters():
    rows = make_rows(trials=3, scenarios=("fire_stage6", "fire_stage40"))
    assert sel_row(paired_summary(rows, "margin", REF, "all"))["n_clusters"] == 6


def test_holm_adjust_known_values_and_nan_handling():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = holm_adjust([0.5, float("nan"), 0.9])
    assert math.isnan(out[1]) and out[2] == 1.0


def test_summary_rows_carry_holm_adjusted_p_values():
    rows = make_rows(gain_conc=0.3, gain_dist=0.3)
    extra = [{**r, "planner": "aco_retention", "margin": r["margin"] + 0.05 * (1 + r["trial"] % 3)}
             for r in rows if r["planner"] == SEL]
    out = [r for r in paired_summary(rows + extra, "margin", REF, "all") if "t_p" in r]
    assert len(out) == 2
    for r in out:
        assert r["t_p_holm"] >= r["t_p"] - 1e-12 and r["t_p_holm"] <= 1.0


@pytest.mark.filterwarnings("ignore:Precision loss")
def test_interaction_detects_a_concentrated_only_advantage():
    res = change_interaction(make_rows(gain_conc=0.3, gain_dist=0.0), "margin", SEL, REF, "locality")
    assert res["n_clusters"] == 6
    assert res["mean_interaction"] == pytest.approx(-0.3, abs=0.03)
    assert res["ci_hi"] < 0 and "concentrated" in res["reading"] and res["trials_favouring_a"] == 6


def test_interaction_reports_no_difference_when_gains_are_equal():
    res = change_interaction(make_rows(gain_conc=0.2, gain_dist=0.2), "margin", SEL, REF, "locality")
    assert abs(res["mean_interaction"]) < 0.03 and "no clear difference" in res["reading"]


def test_interaction_by_fraction_uses_the_split():
    rows = []
    for r in make_rows(gain_conc=0.0, gain_dist=0.0, jitter=True):
        if r["planner"] == SEL and r["target_fraction"] <= 0.4:
            r = {**r, "margin": r["margin"] - 0.4}
        rows.append(r)
    res = change_interaction(rows, "margin", SEL, REF, "fraction", fraction_split=0.4)
    assert res["label_a"] == "fraction<=0.4" and res["mean_interaction"] == pytest.approx(-0.4, abs=0.05)
    assert res["ci_hi"] < 0


def test_interaction_skips_clusters_missing_one_side_and_rejects_bad_factor():
    rows = [r for r in make_rows(trials=4) if not (r["trial"] == 0 and r["locality"] == "distributed")]
    assert change_interaction(rows, "margin", SEL, REF, "locality")["n_clusters"] == 3
    with pytest.raises(ValueError):
        change_interaction(rows, "margin", SEL, REF, "nonsense")


def test_analysis_script_prints_the_interaction_section(tmp_path):
    path = tmp_path / "r.csv"
    rows = make_rows(gain_conc=0.3)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    script = Path(__file__).resolve().parents[1] / "scripts" / "controlled_change" / "analyse_change_experiment.py"
    out = subprocess.run([sys.executable, str(script), str(path), "--metric", "margin", "--group", "locality"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "relatively better" in out.stdout and "n" in out.stdout
