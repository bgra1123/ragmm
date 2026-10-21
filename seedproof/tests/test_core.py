import numpy as np
import pytest

from seedproof import StabilityAdapter, compare_stability, load_german_credit, polish_logistic, stability_check
from seedproof.polish import logistic_objective, platt_apply, platt_fit

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


def data(n=400, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ w))).astype(int)
    return X, y


@pytest.mark.parametrize("step", ["line_search", "entropic", "entropic_exact", "newton"])
def test_polish_lowers_objective_and_reaches_optimum(step):
    X, y01 = data()
    F = np.column_stack([X, np.ones(len(X))])
    y = 2.0 * y01 - 1
    w_star, _ = polish_logistic(np.zeros(F.shape[1]), F, y, 0.01, step="newton")
    w0 = w_star + np.random.default_rng(1).normal(scale=0.5, size=len(w_star))
    w, info = polish_logistic(w0, F, y, 0.01, epochs=60, step=step)
    assert info["objective_after"] < info["objective_before"]
    assert logistic_objective(w, F, y, 0.01) - logistic_objective(w_star, F, y, 0.01) < 1e-4


def test_polish_rejects_unknown_step():
    with pytest.raises(ValueError):
        polish_logistic(np.zeros(2), np.ones((3, 2)), np.ones(3), 0.1, step="bogus")


def test_entropic_step_is_more_cautious_with_larger_theta():
    X, y01 = data(seed=3)
    F = np.column_stack([X, np.ones(len(X))])
    y = 2.0 * y01 - 1
    _, i1 = polish_logistic(np.zeros(F.shape[1]), F, y, 0.01, epochs=1, step="entropic", theta=1.0)
    _, i5 = polish_logistic(np.zeros(F.shape[1]), F, y, 0.01, epochs=1, step="entropic", theta=20.0)
    assert i5["mean_step_times_L"] < i1["mean_step_times_L"]


def test_polish_strategy_removes_seed_variation_for_sgd():
    X, y = data(seed=2)
    make = lambda s: SGDClassifier(loss="log_loss", alpha=0.01, max_iter=3, tol=None, random_state=s)  # noqa: E731
    before = stability_check(make, X, y, X, y, seeds=6)
    after = stability_check(lambda s: StabilityAdapter(make(s), random_state=s), X, y, X, y, seeds=6)
    assert after.summary["disagreement"] < 0.2 * max(before.summary["disagreement"], 1e-3) + 1e-9
    a = StabilityAdapter(make(0)).fit(X, y)
    assert a.strategy_ == "polish" and a.predict(X).shape == (len(y),)
    assert set(np.unique(a.predict(X))) <= {0, 1}


def test_average_strategy_for_random_forest_and_calibration():
    X, y = data(seed=4)
    make = lambda s: RandomForestClassifier(n_estimators=10, max_features=1, random_state=s)  # noqa: E731
    a = StabilityAdapter(make(0), n_seeds=3, calibrate=True, cv=3).fit(X, y)
    assert a.strategy_ == "average" and a.info_["fits"] == 3 * 4
    p = a.predict_proba(X)
    assert p.shape == (len(y), 2) and np.allclose(p.sum(1), 1)
    before = stability_check(make, X, y, X, y, seeds=5)
    after = stability_check(lambda s: StabilityAdapter(make(s), n_seeds=5, random_state=s), X, y, X, y, seeds=5)
    assert after.summary["disagreement"] < before.summary["disagreement"]


def test_platt_roundtrip():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 2000)
    y = (rng.random(2000) < p).astype(float)
    ab = platt_fit(p, y)
    assert abs(ab[0] - 1) < 0.25 and abs(ab[1]) < 0.25
    assert np.all((platt_apply(p, ab) > 0) & (platt_apply(p, ab) < 1))


def test_stability_check_groups_and_reports():
    X, y = data(seed=6)
    make = lambda s: SGDClassifier(loss="log_loss", alpha=0.01, max_iter=2, tol=None, random_state=s)  # noqa: E731
    groups = {"x0 > 1": X[:, 0] > 1, "tiny": np.arange(len(y)) < 3}
    r = stability_check(make, X, y, X, y, seeds=4, groups=groups, name="sgd")
    assert "x0 > 1" in r.groups and "tiny" not in r.groups
    s = r.summary
    assert 0 <= s["disagreement"] <= 1 and s["retrains"] == 4
    assert abs(s["wrong"] - s["wrong_majority"] - s["wrong_excess"]) < 1e-12
    assert "| sgd |" in compare_stability([r]) and "## sgd" in r.to_markdown()
    assert isinstance(r.statements(), list)
    with pytest.raises(ValueError):
        stability_check(make, X, y, X, y, seeds=1)


def test_bundled_german_credit():
    df = load_german_credit()
    assert df.shape == (1000, 21) and (df.creditability == "good").sum() == 700


def test_cli_adapter(tmp_path):
    from seedproof.cli import main
    out = tmp_path / "s.md"
    from seedproof.datasets import DATA
    assert main([str(DATA / "german_credit.csv"), "--target", "creditability", "--positive", "good", "--model", "sgd",
                 "--seeds", "3", "--group-columns", "housing", "--report", str(out)]) == 0
    text = out.read_text()
    assert "sgd + adapter" in text and "housing = for free" in text


@pytest.mark.parametrize("step", ["line_search", "entropic", "entropic_exact"])
@pytest.mark.parametrize("theta", [1.0, 5.0])
def test_streaming_matches_in_memory(step, theta):
    from seedproof import iter_batches, polish_logistic_streaming
    X, y01 = data(n=300, d=4, seed=7)
    F = np.column_stack([X, np.ones(len(X))])
    y = 2.0 * y01 - 1
    mask = np.r_[np.ones(4), 0.0]
    w0 = np.random.default_rng(1).normal(scale=0.3, size=5)
    w_ref, i_ref = polish_logistic(w0, F, y, 0.01, epochs=4, step=step, theta=theta, mask=mask)
    for search, tol in (("newton", 1e-6), ("grid", 5e-3)):
        outs = []
        for bs in (13, 300):
            w, info = polish_logistic_streaming(w0, iter_batches(F, y, bs), 0.01, epochs=4, step=step, theta=theta,
                                                mask=mask, search=search)
            assert np.abs(w - w_ref).max() < tol
            assert abs(info["objective_after"] - i_ref["objective_after"]) < 1e-4
            outs.append(w)
        assert np.abs(outs[0] - outs[1]).max() < 1e-9           # the batch size does not change the result


def test_streaming_rejects_bad_step_and_bad_search():
    from seedproof import iter_batches, polish_logistic_streaming
    F, y = np.ones((4, 2)), np.array([1.0, -1.0, 1.0, -1.0])
    with pytest.raises(ValueError):
        polish_logistic_streaming(np.zeros(2), iter_batches(F, y, 2), 0.1, step="bogus")
    with pytest.raises(ValueError):
        polish_logistic_streaming(np.zeros(2), iter_batches(F, y, 2), 0.1, search="bogus")


def test_adapter_batch_size_matches_unbatched():
    X, y = data(seed=9)
    make = lambda: SGDClassifier(loss="log_loss", alpha=0.01, max_iter=3, tol=None, random_state=0)  # noqa: E731
    a = StabilityAdapter(make(), step="entropic").fit(X, y)
    b = StabilityAdapter(make(), step="entropic", batch_size=64).fit(X, y)
    assert np.abs(a.fitted_["coef"] - b.fitted_["coef"]).max() < 1e-5


def test_streaming_newton_matches_in_memory_and_solves_exactly():
    from seedproof import iter_batches, polish_logistic_streaming
    X, y01 = data(n=300, d=4, seed=11)
    F = np.column_stack([X, np.ones(len(X))])
    y = 2.0 * y01 - 1
    mask = np.r_[np.ones(4), 0.0]
    w0 = np.random.default_rng(2).normal(scale=0.5, size=5)
    w_ref, _ = polish_logistic(w0, F, y, 0.01, step="newton", mask=mask)
    for bs in (17, 300):
        w, info = polish_logistic_streaming(w0, iter_batches(F, y, bs), 0.01, epochs=30, step="newton", mask=mask)
        assert np.abs(w - w_ref).max() < 1e-9
        assert info["iterations"] <= 10 and info["passes"] == 2 * info["iterations"] + 1
    # different starting points reach the same model: no seed dependence
    ws = [polish_logistic_streaming(np.random.default_rng(s).normal(size=5), iter_batches(F, y, 50), 0.01, epochs=30,
                                    step="newton", mask=mask)[0] for s in range(3)]
    assert max(np.abs(a - b).max() for a in ws for b in ws) < 1e-8
