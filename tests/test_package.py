"""Tests for the tabular package API: encoder, classifier, tail report and CLI."""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from risk_averse_tuning import RAGMMBounds, RiskAverseClassifier, TabularEncoder, tail_report
from risk_averse_tuning.classifier import batch_indices, choose_parameters, objective, smoothness, solve_exact, train_gmm
from risk_averse_tuning.cli import main as cli_main
from risk_averse_tuning.tabular import binary_target, stratified_split
from risk_averse_tuning.tail_report import empirical_evar, leverage

ROOT = pathlib.Path(__file__).resolve().parents[1]
GERMAN = ROOT / "research" / "model_multiplicity" / "data" / "german_credit.csv"
PROTECTED = ["personal_status_and_sex", "foreign_worker", "age_in_years"]


def toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"income": rng.normal(50, 10, n), "debt": rng.normal(20, 5, n),
                       "region": rng.choice(["north", "south", "east"], n),
                       "flag": rng.choice([True, False], n)})
    logit = 0.15 * (df.income - 50) - 0.3 * (df.debt - 20) + (df.region == "north") * 1.0
    df["label"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    return df


# ------------------------------ encoder ------------------------------ #

def test_encoder_handles_missing_unseen_and_standardizes():
    df = toy()
    df.loc[3, "income"] = np.nan
    df.loc[4, "region"] = None
    enc = TabularEncoder(drop=["label"]).fit(df)
    X = enc.transform(df)
    assert X.shape[1] == len(enc.feature_names_) and enc.feature_names_[-1] == "(intercept)"
    assert np.allclose(X[:, :-1].mean(0), 0, atol=1e-12) and np.all(X[:, -1] == 1)
    assert "region=<missing>" in enc.feature_names_ and not np.isnan(X).any()
    new = df.head(2).copy()
    new["region"] = ["west", "north"]                               # unseen category -> zero indicators
    Xn = enc.transform(new)
    cols = [i for i, f in enumerate(enc.feature_names_) if f.startswith("region=")]
    assert np.allclose(Xn[0, cols], ((0 - enc.mean_) / enc.std_)[cols])
    with pytest.raises(ValueError):
        enc.transform(df.drop(columns=["debt"]))


def test_binary_target_and_split():
    y, pos = binary_target(pd.Series(["a", "b", "a", "b"]), "b")
    assert pos == "b" and list(y) == [-1, 1, -1, 1]
    y2, pos2 = binary_target(pd.Series([0, 1, 1, 0]), "1")
    assert pos2 == 1 and list(y2) == [-1, 1, 1, -1]
    with pytest.raises(ValueError):
        binary_target(pd.Series([1, 2, 3]))
    yy = np.r_[np.ones(70), -np.ones(30)]
    m = stratified_split(yy, 0.3, 0)
    assert m.sum() == 30 and (yy[m] > 0).sum() == 21


# ------------------------------ classifier ------------------------------ #

def test_ragmm_parameters_are_the_papers_problem_51():
    df = toy()
    y, _ = binary_target(df.label, "yes")
    X = TabularEncoder(drop=["label"]).fit_transform(df)
    params, info = choose_parameters("ragmm", X, y, 0.01, 500, 16)
    sol = RAGMMBounds(zeta=0.05).solve(smoothness(X, 0.01), 0.01, X.shape[1], rate_slack=0.25)
    assert (params["alpha"], params["beta"], params["gamma"]) == (sol.alpha, sol.beta, sol.gamma)
    assert info["rate_ratio"] <= 1.25 + 1e-12


def test_package_reproduces_research_parameters_on_german_credit():
    df = pd.read_csv(GERMAN)
    y, _ = binary_target(df.creditability, "good")
    t = stratified_split(y, 0.3, 0)
    f = df.drop(columns=["creditability"])
    X = TabularEncoder(drop=PROTECTED).fit(f[~t]).transform(f[~t])
    assert smoothness(X, 0.01) == pytest.approx(0.9888127781381304, rel=1e-10)
    p, _ = choose_parameters("ragmm", X, y[~t], 0.01, 2000, 16)
    assert p["alpha"] == pytest.approx(0.010011886330329817, rel=1e-9) and p["beta"] == pytest.approx(0.98008798844846, rel=1e-9)


def test_train_gmm_matches_torch_library_gmm():
    torch = pytest.importorskip("torch")
    from risk_averse_tuning import GMM
    df = toy(200)
    y, _ = binary_target(df.label, "yes")
    X = TabularEncoder(drop=["label"]).fit_transform(df)
    idx = batch_indices([7], 40, len(y), 8)
    a, b, g = 0.2, 0.6, 0.3
    W = train_gmm(X, y, 0.01, idx, a, b, g)[0]
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = GMM([w], lr=a, beta=b, gamma=g)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for k in range(40):
        bi = torch.tensor(idx[0, k])

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.softplus(-(Xt[bi] @ w) * yt[bi]).mean() + 0.005 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
    assert np.abs(W - w.detach().numpy()).max() < 1e-10


def test_classifier_fits_predicts_and_is_reproducible():
    df = toy()
    y, _ = binary_target(df.label, "yes")
    X = TabularEncoder(drop=["label"]).fit_transform(df)
    w_star, f_star = solve_exact(X, y, 0.01)
    for method in ("ragmm", "sgd", "careful"):
        c1 = RiskAverseClassifier(method=method, steps=600, seed=3).fit(X, y)
        c2 = RiskAverseClassifier(method=method, steps=600, seed=3).fit(X, (y + 1) / 2)   # {0,1} labels accepted
        assert np.array_equal(c1.coef_, c2.coef_)
        assert objective(c1.coef_, X, y, 0.01)[0] - f_star < 0.2
        assert (c1.predict(X) == y).mean() > 0.6
        P = c1.predict_proba(X)
        assert np.allclose(P.sum(1), 1) and P.shape == (len(y), 2)
    with pytest.raises(ValueError):
        RiskAverseClassifier(method="adam")
    with pytest.raises(ValueError):
        RiskAverseClassifier(lam=0)


def test_careful_is_tail_averaged_sgd():
    df = toy()
    y, _ = binary_target(df.label, "yes")
    X = TabularEncoder(drop=["label"]).fit_transform(df)
    p, info = choose_parameters("careful", X, y, 0.01, 300, 16)
    assert p["beta"] == p["gamma"] == 0 and p["tail_average"] and info["tuning_runs"] == 240


# ------------------------------ tail report ------------------------------ #

def test_empirical_evar_is_above_cvar_and_mean():
    x = np.random.default_rng(0).exponential(size=5000)
    cvar = np.sort(x)[-250:].mean()
    ev = empirical_evar(x, 0.05)
    assert ev >= cvar - 1e-9 and ev <= x.max() + 1e-9


def test_leverage_matches_definition():
    df = toy(150)
    y, _ = binary_target(df.label, "yes")
    X = TabularEncoder(drop=["label"]).fit_transform(df)
    w, _ = solve_exact(X, y, 0.01)
    h = leverage(X, y, w, 0.01, X[:3])
    p = 1 / (1 + np.exp(-X @ w))
    s = 1 / (1 + np.exp(-y * (X @ w)))
    H = sum(s[i] * (1 - s[i]) * np.outer(X[i], X[i]) for i in range(len(y))) / len(y) + 0.01 * np.eye(X.shape[1])
    ref = [p[i] * (1 - p[i]) * X[i] @ np.linalg.solve(H, X[i]) / len(y) for i in range(3)]
    assert np.allclose(h, ref)


def test_tail_report_contents_and_consistency():
    df = toy(300)
    rep = tail_report(df, "label", positive="yes", retrains=40, steps=300)
    d = rep.to_dict()
    assert set(d["methods"]) == {"ragmm", "sgd", "careful"}
    for m in d["methods"].values():
        assert 0 <= m["changed_mean"] <= rep.summary["n_test"]
        assert m["changed_worst5"] >= m["changed_mean"] - 1e-9
        assert m["coin_flip_rows"] <= m["ever_changed_rows"] <= rep.summary["n_test"]
        assert m["objective_evar"] >= m["objective_cvar5"] - 1e-9 >= m["objective_mean"] - 1e-9
    assert len(rep.at_risk_rows) == round(0.1 * rep.summary["n_test"])
    md = rep.to_markdown()
    assert "not a finding of bias" in md and "Rows at risk" in md
    with pytest.raises(ValueError):
        tail_report(df, "label", methods=["adam"])
    with pytest.raises(ValueError):
        tail_report(df, "label", retrains=5)


def test_tail_report_reproduces_research_on_german_credit():
    df = pd.read_csv(GERMAN)
    rep = tail_report(df, "creditability", positive="good", drop=PROTECTED, methods=["sgd"], retrains=200)
    # research/report/credit_story.json (1,000 retrains): 55.16 changes per retrain, AUC 0.762
    assert rep.summary["exact_auc"] == pytest.approx(0.761957671957672, abs=1e-9)
    assert rep.methods["sgd"]["changed_mean"] == pytest.approx(55.2, abs=1.5)


# ------------------------------ CLI ------------------------------ #

def test_cli_end_to_end(tmp_path):
    csv = tmp_path / "toy.csv"
    toy(250).to_csv(csv, index=False)
    model, report = tmp_path / "m.json", tmp_path / "r.md"
    rc = cli_main([str(csv), "--target", "label", "--positive", "yes", "--steps", "300", "--model", str(model),
                   "--report", str(report), "--retrains", "20"])
    assert rc == 0
    m = json.loads(model.read_text())
    assert len(m["coefficients"]) == len(m["features"]) and m["method"] == "ragmm"
    assert report.read_text().startswith("# Tail report") and report.with_suffix(".json").exists()


def test_group_risk_statements_and_consistency():
    df = toy(400)
    df["age"] = np.random.default_rng(1).integers(20, 70, len(df))
    rep = tail_report(df, "label", positive="yes", methods=["sgd", "careful"], retrains=40, steps=300, drop=["age"],
                      groups={"age under 30": lambda d: d.age < 30}, group_columns=["region"], min_group_size=10)
    for m, res in rep.group_risk.items():
        allr = res["All held-out rows"]
        assert allr["rows"] == rep.summary["n_test"]
        assert allr["p_wrong"] == pytest.approx(allr["p_wrong_exact"] + allr["excess"])
        assert 0 <= allr["p_wrong"] <= 1 and allr["p_wrong_ci95"][0] <= allr["p_wrong"] <= allr["p_wrong_ci95"][1] + 1e-9
        assert "age under 30" in res and any(k.startswith("region = ") for k in res)
        # groups partition by region: row counts add up
        assert sum(r["rows"] for k, r in res.items() if k.startswith("region = ")) <= rep.summary["n_test"]
    md = rep.to_markdown()
    assert "Chance of a wrong decision, by type of row" in md and "randomness of training" in md
    for sent in rep.statements(min_rows=10):
        assert "gives a wrong decision" in sent
    with pytest.raises(ValueError):
        tail_report(df, "label", retrains=20, group_columns=["nope"])


def test_group_risk_matches_direct_count_for_exact_model_floor():
    df = toy(300)
    rep = tail_report(df, "label", positive="yes", methods=["careful"], retrains=20, steps=400)
    r = rep.group_risk["careful"]["All held-out rows"]
    assert r["p_wrong_exact"] == pytest.approx(1 - rep.summary["exact_accuracy"])
