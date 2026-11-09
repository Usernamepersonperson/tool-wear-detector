import numpy as np
import pytest

from toolwear import features
from toolwear.features import FS


def test_band_energy_sine_lands_in_its_band():
    t = np.arange(4000) / FS
    x = np.sin(2 * np.pi * 60 * t)
    in_band = features.band_energy(x, FS, 40, 80)
    other = features.band_energy(x, FS, 1, 40)
    assert in_band > 50 * other


def test_band_energy_total_power_matches_variance():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2.0, 8000)
    total = features.band_energy(x, FS, 0, FS)
    assert total == pytest.approx(x.var(), rel=0.1)


def test_band_energy_too_short():
    with pytest.raises(ValueError):
        features.band_energy(np.zeros(5), FS, 1, 40)


def test_band_energy_empty_band_is_zero():
    assert features.band_energy(np.random.default_rng(0).normal(size=512), FS, 500, 600) == 0.0


def test_rms_of_sine_is_amplitude_over_root_two():
    x = 3.0 * np.sin(2 * np.pi * 10 * np.arange(5000) / FS)
    f = features.channel_features(x)
    assert 10 ** f["rms"] == pytest.approx(3.0 / np.sqrt(2), rel=0.01)


def test_kurtosis_of_gaussian_is_near_zero():
    x = np.random.default_rng(2).normal(size=20000)
    assert features.channel_features(x)["kurt"] == pytest.approx(0.0, abs=0.15)


def test_kurtosis_rises_with_impulses():
    rng = np.random.default_rng(3)
    x = rng.normal(size=3000)
    y = x.copy()
    y[::300] += 25
    assert features.channel_features(y)["kurt"] > features.channel_features(x)["kurt"] + 5


def test_crest_of_square_wave_is_one():
    x = np.tile([1.0, -1.0], 500)
    assert features.channel_features(x)["crest"] == pytest.approx(0.0, abs=1e-6)


def test_channel_features_rejects_nan_and_short():
    with pytest.raises(ValueError):
        features.channel_features(np.array([1.0, np.nan] * 20))
    with pytest.raises(ValueError):
        features.channel_features(np.ones(4))


def test_constant_signal_does_not_crash():
    f = features.channel_features(np.full(200, 3.0))
    assert f["kurt"] == 0.0
    assert np.isfinite(list(f.values())).all()


def test_cut_window_finds_the_cut():
    x = np.concatenate([np.full(500, 1.0), np.full(3000, 8.0), np.full(500, 1.0)])
    w = features.cut_window(x)
    assert (w.start, w.stop) == (800, 3200)  # cut 500..3500 minus 10 percent at each end


def test_cut_window_flat_signal_falls_back_to_all():
    w = features.cut_window(np.full(1000, 2.0))
    assert (w.start, w.stop) == (0, 1000)


def test_cut_window_short_signal():
    w = features.cut_window(np.arange(10.0))
    assert (w.start, w.stop) == (0, 10)


def test_run_features_length_and_names(table):
    names = features.feature_names()
    assert len(names) == len(features.CHANNELS) * len(features.STATS) == 42
    assert table.feats.shape[1] == 42
    assert np.isfinite(table.feats).all()


def test_wear_raises_load_feature():
    rng = np.random.default_rng(4)
    from conftest import synth_signal

    names = features.feature_names()
    lo = features.run_features(synth_signal(0.1, rng))
    hi = features.run_features(synth_signal(0.6, rng))
    k = names.index("smcDC_mean_abs")
    assert hi[k] > lo[k]


def test_relative_to_baseline():
    d = features.relative_to_baseline(np.array([3.0, 5.0]), np.array([1.0, 1.0]))
    assert d.tolist() == [2.0, 4.0]


def test_power_trend_trailing_average():
    out = features.power_trend(np.array([1.0, 2.0, 3.0, 4.0]), window=2)
    assert out.tolist() == [1.0, 1.5, 2.5, 3.5]


def test_power_trend_bad_window():
    with pytest.raises(ValueError):
        features.power_trend(np.ones(3), window=0)
