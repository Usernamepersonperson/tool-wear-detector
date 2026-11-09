import numpy as np
import pytest

from toolwear import alarm, features, policy, rul

# ---- rul ----


def test_smooth_wear_is_monotone():
    out = rul.smooth_wear(np.array([0.1, 0.3, 0.2, 0.5, 0.1]))
    assert np.all(np.diff(out) >= 0)


def test_smooth_wear_alpha_one_is_running_max():
    out = rul.smooth_wear(np.array([0.1, 0.3, 0.2]), alpha=1.0)
    assert out.tolist() == [0.1, 0.3, 0.3]


def test_smooth_wear_rejects_bad_alpha_and_handles_empty():
    with pytest.raises(ValueError):
        rul.smooth_wear(np.array([1.0]), alpha=0.0)
    assert rul.smooth_wear(np.array([])).size == 0


def test_time_to_limit_interpolates():
    t = np.array([0.0, 10.0, 20.0])
    v = np.array([0.0, 0.2, 0.6])
    assert rul.time_to_limit(t, v, 0.4) == pytest.approx(15.0)


def test_time_to_limit_skips_nan_and_none_when_never():
    t = np.array([0.0, 5.0, 10.0])
    v = np.array([0.0, np.nan, 0.2])
    assert rul.time_to_limit(t, v, 0.4) is None
    assert rul.time_to_limit(t, v, 0.1) == pytest.approx(5.0)


def test_time_to_limit_first_point_already_over():
    assert rul.time_to_limit(np.array([2.0, 4.0]), np.array([0.5, 0.6]), 0.4) == 2.0


def test_estimate_rul_linear_trend_known_value():
    t = np.arange(0.0, 20.0, 2.0)
    w = 0.02 * t  # reaches 0.4 at t = 20
    assert rul.estimate_rul(t, w, 0.4) == pytest.approx(20.0 - t[-1], abs=1e-6)


def test_estimate_rul_edge_cases():
    t = np.arange(5.0)
    assert np.isnan(rul.estimate_rul(t[:2], np.array([0.1, 0.2]), 0.4))
    assert rul.estimate_rul(t, np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 0.4) == 0.0
    assert rul.estimate_rul(t, np.full(5, 0.1), 0.4) == rul.RUL_CAP
    with pytest.raises(ValueError):
        rul.estimate_rul(t, np.ones(4), 0.4)


def test_estimate_rul_is_capped():
    t = np.arange(5.0)
    assert rul.estimate_rul(t, 1e-6 * t + 0.1, 0.4, cap=50.0) == 50.0


# ---- alarm ----


def test_threshold_at_fpr_bounds_false_alarms():
    neg = np.arange(100.0)
    thr = alarm.threshold_at_fpr(neg, 0.05)
    assert alarm.false_alarm_rate(neg, thr) <= 0.05
    assert alarm.false_alarm_rate(neg, thr) >= 0.04


def test_threshold_at_zero_fpr_is_above_all_negatives():
    neg = np.array([1.0, 2.0, 3.0])
    assert alarm.threshold_at_fpr(neg, 0.0) > 3.0


def test_threshold_rejects_bad_input():
    with pytest.raises(ValueError):
        alarm.threshold_at_fpr(np.array([]), 0.05)
    with pytest.raises(ValueError):
        alarm.threshold_at_fpr(np.array([1.0]), 1.0)


def test_recall_at():
    assert alarm.recall_at(np.array([0.1, 0.5, 0.9]), 0.5) == pytest.approx(2 / 3)
    assert np.isnan(alarm.recall_at(np.array([]), 0.5))


def test_first_alarm():
    t = np.array([1.0, 2.0, 3.0])
    assert alarm.first_alarm(t, np.array([0.1, 0.6, 0.7]), 0.5) == 2.0
    assert alarm.first_alarm(t, np.array([0.1, 0.2, 0.3]), 0.5) is None


def test_impulse_score_needs_history():
    assert np.isnan(alarm.impulse_score(np.zeros((2, 42)), np.zeros(42)))


def test_impulse_score_flags_a_jump_not_a_steady_run():
    rng = np.random.default_rng(5)
    from conftest import synth_signal

    hist = np.array([features.run_features(synth_signal(0.2, rng)) for _ in range(5)])
    steady = features.run_features(synth_signal(0.2, rng))
    sig = alarm.inject_breakage(synth_signal(0.2, rng), rng, burst=40.0, step=0.5)
    jumped = features.run_features(sig)
    assert alarm.impulse_score(hist, jumped) > alarm.impulse_score(hist, steady) + 5


def test_inject_breakage_copies_and_adds_energy():
    rng = np.random.default_rng(6)
    from conftest import synth_signal

    base = synth_signal(0.2, rng)
    before = {k: v.copy() for k, v in base.items()}
    out = alarm.inject_breakage(base, rng, burst=20.0, step=0.3)
    for k in base:
        assert np.array_equal(base[k], before[k])
    assert np.std(out["AE_spindle"]) > np.std(base["AE_spindle"])
    assert out["smcDC"].max() > base["smcDC"].max()


# ---- policy ----


def test_counter_life_modes():
    assert policy.counter_life([10, 20, 30], "conservative") == 10
    assert policy.counter_life([10, 20, 30], "mean") == 20
    with pytest.raises(ValueError):
        policy.counter_life([], "mean")
    with pytest.raises(ValueError):
        policy.counter_life([1.0], "median")


def test_recipe_counter_uses_same_recipe_and_factor():
    known = [((1, 1.5, 0.5), 10.0), ((1, 1.5, 0.5), 20.0), ((2, 0.75, 0.25), 100.0)]
    assert policy.recipe_counter(known, (1, 1.5, 0.5)) == 15.0
    assert policy.recipe_counter(known, (1, 1.5, 0.5), 0.8) == pytest.approx(12.0)


def test_recipe_counter_falls_back_to_global_mean():
    known = [((1, 1.0, 1.0), 10.0), ((2, 1.0, 1.0), 30.0)]
    assert policy.recipe_counter(known, (9, 9, 9)) == 20.0


def test_recipe_counter_validation():
    with pytest.raises(ValueError):
        policy.recipe_counter([], (1, 1, 1))
    with pytest.raises(ValueError):
        policy.recipe_counter([((1, 1, 1), 5.0)], (1, 1, 1), factor=0.0)


def test_recommend_change_first_crossing():
    w = np.array([0.1, 0.2, 0.3, 0.3])
    assert policy.recommend_change(np.arange(4.0), w, 0.3) == 2
    assert policy.recommend_change(np.arange(4.0), w, 0.9) is None


def test_change_time_censored_returns_last_time():
    t = np.array([3.0, 6.0, 9.0])
    assert policy.change_time(t, np.array([0.1, 0.2, 0.3]), 0.25) == (9.0, True)
    assert policy.change_time(t, np.array([0.1, 0.2, 0.3]), 0.9) == (9.0, False)


def test_outcome_late_and_early():
    late = policy.outcome(30.0, 25.0, 3.0)
    assert late["late"] == 1.0 and late["overrun_min"] == 5.0 and late["unused_min"] == 0.0
    early = policy.outcome(20.0, 25.0, 5.0)
    assert early["late"] == 0.0 and early["unused_min"] == 5.0 and early["parts"] == 4.0


def test_outcome_rejects_bad_part_time():
    with pytest.raises(ValueError):
        policy.outcome(1.0, 2.0, 0.0)
