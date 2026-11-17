"""Generate report assets: smartwatch feasibility table + appendix confusion matrices.

Run from the coursework root (the folder containing 'hrv dataset/').
Outputs land in hrv dataset/data/processed/ and .../figures/appendix/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROC = Path("hrv dataset/data/processed")
APPFIG = PROC / "figures/appendix"
APPFIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="white")

TIME_DOM = ["MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD", "HR",
            "pNN25", "pNN50", "SD1", "SD2", "KURT", "SKEW"]
FREQ_DOM = ["VLF", "LF", "HF", "TP", "LF_HF", "HF_LF", "LF_nu", "HF_nu"]
ALL_HRV = TIME_DOM + FREQ_DOM
BIN = ["non-stress", "stress"]


def load():
    H = pd.read_csv(PROC / "features_hrv_windows.csv")
    Hz = H.copy()
    for c in ALL_HRV:
        Hz[c + "_z"] = Hz.groupby("subject")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    lab = pd.read_csv(PROC / "features_engineered.csv")
    for c in ["HR", "RMSSD"]:
        lab[c + "_z"] = lab.groupby("subject")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    return H, Hz, lab


def z(cols):
    return [c + "_z" for c in cols]


MODELS = {
    "Dummy": lambda: DummyClassifier(strategy="most_frequent"),
    "GNB": lambda: GaussianNB(),
    "LogReg": lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
    "RF": lambda: RandomForestClassifier(n_estimators=400, max_depth=10,
                                         class_weight="balanced", random_state=42, n_jobs=-1),
}


def smartwatch_table(Hz):
    setups = {
        "S1 lab: full HRV + SCL (ECG chest strap + EDA sensor)": z(ALL_HRV) + ["SCL_z"],
        "S2 watch-plausible: full HRV, no SCL": z(ALL_HRV),
        "S3 watch-realistic: time-domain HRV only, no SCL": z(TIME_DOM),
        "S4 watch-minimal: HR + RMSSD + SDNN + pNN50": z(["HR", "RMSSD", "SDRR", "pNN50"]),
        "S5 raw baseline: HR + RMSSD only (no personalisation)": ["HR", "RMSSD"],
    }
    rows = []
    y, g = Hz.is_stress.to_numpy(), Hz.subject.to_numpy()
    for tag, cols in setups.items():
        X = Hz[cols].to_numpy()
        for name, ctor in MODELS.items():
            accs, f1s = [], []
            for tr, te in GroupKFold(n_splits=5).split(X, y, g):
                sc = StandardScaler()
                p = ctor().fit(sc.fit_transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
                accs.append(accuracy_score(y[te], p))
                f1s.append(f1_score(y[te], p, zero_division=0))
            rows.append({"sensor_setup": tag, "n_features": len(cols), "model": name,
                         "acc": np.mean(accs), "acc_sd": np.std(accs), "f1": np.mean(f1s)})
    R = pd.DataFrame(rows)
    R.to_csv(PROC / "smartwatch_feasibility.csv", index=False)
    print(R.pivot_table(index=["sensor_setup", "n_features"], columns="model",
                        values="acc").round(3).to_string())
    return R


def _oof(df, cols, ctor, y_col="is_stress"):
    X, y, g = df[cols].to_numpy(), df[y_col].to_numpy(), df.subject.to_numpy()
    return y, cross_val_predict(make_pipeline(StandardScaler(), ctor()), X, y,
                                groups=g, cv=GroupKFold(n_splits=5))


def _cm(ax, y, p, labels, title, cmap="Blues"):
    cm = confusion_matrix(y, p, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=False,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(f"{title}\nacc={accuracy_score(y, p):.3f}  "
                 f"macro-F1={f1_score(y, p, average='macro', zero_division=0):.3f}", fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")


def confusion_figures(Hz, lab):
    # A1 — model families
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, nm in zip(axes, ["GNB", "LogReg", "RF"]):
        y, p = _oof(Hz, z(ALL_HRV) + ["SCL_z"], MODELS[nm])
        _cm(ax, y, p, [0, 1], f"{nm} — full HRV+SCL")
        ax.set_xticklabels(BIN); ax.set_yticklabels(BIN)
    plt.tight_layout(); plt.savefig(APPFIG / "A1_confusion_binary_models.png", dpi=150); plt.close()

    # A2 — sensor setups
    setups = [("Lab: full HRV + SCL (22f)", z(ALL_HRV) + ["SCL_z"]),
              ("Watch: time-domain HRV (13f)", z(TIME_DOM)),
              ("Watch-minimal: 4 features", z(["HR", "RMSSD", "SDRR", "pNN50"])),
              ("Raw HR+RMSSD, no personalisation", ["HR", "RMSSD"])]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (nm, cols) in zip(axes, setups):
        y, p = _oof(Hz, cols, MODELS["LogReg"])
        _cm(ax, y, p, [0, 1], nm, cmap="Greens")
        ax.set_xticklabels(BIN); ax.set_yticklabels(BIN)
    plt.tight_layout(); plt.savefig(APPFIG / "A2_confusion_sensor_setups.png", dpi=150); plt.close()

    # A3 — 4-class on the full labelled table
    order = ["rest", "no stress", "time pressure", "interruption"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, nm in zip(axes, ["GNB", "RF"]):
        X = lab[["HR_z", "RMSSD_z", "SCL_z"]].to_numpy()
        y, g = lab["label"].to_numpy(), lab.subject.to_numpy()
        p = cross_val_predict(make_pipeline(StandardScaler(), MODELS[nm]()), X, y,
                              groups=g, cv=GroupKFold(n_splits=5))
        _cm(ax, y, p, order, f"{nm} — 4-class, 3 physiological features", cmap="Oranges")
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout(); plt.savefig(APPFIG / "A3_confusion_4class.png", dpi=150); plt.close()

    # A4 — leakage
    X, y, g = Hz[z(ALL_HRV)].to_numpy(), Hz.is_stress.to_numpy(), Hz.subject.to_numpy()
    pipe = lambda: make_pipeline(StandardScaler(),
                                 RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
    p_rand = cross_val_predict(pipe(), X, y, cv=StratifiedKFold(10, shuffle=True, random_state=42))
    p_grp = cross_val_predict(pipe(), X, y, groups=g, cv=GroupKFold(n_splits=5))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    _cm(axes[0], y, p_rand, [0, 1], "Random 10-fold CV (LEAKY)", cmap="Reds")
    _cm(axes[1], y, p_grp, [0, 1], "Subject-grouped 5-fold CV (honest)")
    for ax in axes:
        ax.set_xticklabels(BIN); ax.set_yticklabels(BIN)
    plt.tight_layout(); plt.savefig(APPFIG / "A4_confusion_leakage.png", dpi=150); plt.close()
    print(f"A4: random={accuracy_score(y, p_rand):.3f}  grouped={accuracy_score(y, p_grp):.3f}")


if __name__ == "__main__":
    H, Hz, lab = load()
    smartwatch_table(Hz)
    confusion_figures(Hz, lab)
    print("report assets regenerated")
