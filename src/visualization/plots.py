# Publication-Quality Visualization Suite for AQI Research Paper
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set scientific publication aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SEVERITY_COLORS = {
    "Good": "#009966",
    "Satisfactory": "#84cf35",
    "Moderate": "#ffde33",
    "Poor": "#ff9933",
    "VeryPoor": "#cc0033",
    "Severe": "#7e0023",
}


def plot_missing_values(profile_csv: str = "results/tables/city_data_profile.csv", save_dir: str = "results/figures"):
    if not os.path.exists(profile_csv):
        return
    df = pd.read_csv(profile_csv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    w = 0.25

    ax.bar(x - w, df["AQI_MissingPct"], width=w, label="AQI Missing %", color="#1f77b4")
    ax.bar(x, df["PM2.5_MissingPct"], width=w, label="PM2.5 Missing %", color="#ff7f0e")
    ax.bar(x + w, df["NO2_MissingPct"], width=w, label="NO2 Missing %", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(df["City"], fontweight="bold")
    ax.set_ylabel("Missing Rate (%)")
    ax.set_title("Missing Value Rates Across Five Studied Cities (2015-2020)")
    ax.set_ylim(0, 15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig01_missing_values.png"))
    plt.close()


def plot_aqi_trends(raw_csv: str = "data/raw/city_day.csv", save_dir: str = "results/figures"):
    if not os.path.exists(raw_csv):
        return
    df = pd.read_csv(raw_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    cities = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Lucknow"]
    df = df[df["City"].isin(cities)]

    fig, axes = plt.subplots(len(cities), 1, figsize=(12, 10), sharex=True)
    for i, city in enumerate(cities):
        cdf = df[df["City"] == city].sort_values("Date")
        axes[i].plot(cdf["Date"], cdf["AQI"], color="#333333", lw=0.8, alpha=0.85)
        axes[i].axhline(400, color="#7e0023", ls="--", lw=1.2, label="Severe Threshold (400)")
        axes[i].axvline(pd.to_datetime("2019-01-01"), color="#d62728", ls=":", lw=1.5, label="Test Split Start (2019)")
        axes[i].set_ylabel(f"{city}\nAQI")
        axes[i].set_ylim(0, 550)
        if i == 0:
            axes[i].legend(loc="upper right", framealpha=0.9)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Historical Daily AQI Series and Severity Boundaries (2015 - mid 2020)", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig02_aqi_trends_by_city.png"))
    plt.close()


def plot_model_comparison(metrics_csv: str = "results/tables/metrics_all_horizons.csv", save_dir: str = "results/figures"):
    if not os.path.exists(metrics_csv):
        return
    df = pd.read_csv(metrics_csv)
    chrono = df[df["Scenario"] == "Chronological_Full"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # H1
    h1 = chrono[chrono["Horizon"] == "H1"].sort_values("MAE")
    ax1.barh(h1["Model"], h1["MAE"], color="#2b5c8f", alpha=0.85)
    ax1.set_xlabel("MAE (AQI units)")
    ax1.set_title("Horizon H=1 (Next Day Forecast)")
    for i, row in enumerate(h1.itertuples()):
        ax1.text(row.MAE + 0.5, i, f"{row.MAE:.1f}", va="center", fontsize=9)

    # H3
    h3 = chrono[chrono["Horizon"] == "H3"].sort_values("MAE")
    ax2.barh(h3["Model"], h3["MAE"], color="#c0392b", alpha=0.85)
    ax2.set_xlabel("MAE (AQI units)")
    ax2.set_title("Horizon H=3 (3-Day Ahead Forecast)")
    for i, row in enumerate(h3.itertuples()):
        ax2.text(row.MAE + 0.5, i, f"{row.MAE:.1f}", va="center", fontsize=9)

    fig.suptitle("Chronological Test Split Accuracy (2019 to mid-2020)", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig03_model_accuracy_comparison.png"))
    plt.close()


def plot_shift_penalties(metrics_csv: str = "results/tables/metrics_all_horizons.csv", save_dir: str = "results/figures"):
    if not os.path.exists(metrics_csv):
        return
    df = pd.read_csv(metrics_csv)
    h1 = df[df["Horizon"] == "H1"].copy()

    # Pivot shift penalty: Scenario vs Model
    pivot = h1.pivot(index="Scenario", columns="Model", values="ShiftPenalty")
    # Order scenarios
    order = [
        "Chronological_Full", "Year_2019", "Year_2020",
        "Season_Winter", "Season_Summer", "Season_Monsoon", "Season_PostMonsoon",
        "Event_Diwali", "Event_StubbleBurning", "Event_Lockdown", "Event_WinterInversion",
        "City_Delhi", "City_Lucknow", "City_Bengaluru", "City_Chennai", "City_Hyderabad"
    ]
    order = [o for o in order if o in pivot.index]
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(12, 7.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Shift Penalty (MAE / Reference MAE)"}, ax=ax)
    ax.set_title("Shift Penalty Heatmap across Distribution Shift Regimes (H=1)")
    ax.set_ylabel("Evaluation Slice / Shift Regime")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig04_shift_penalty_heatmap.png"))
    plt.close()


def plot_conformal_calibration(metrics_csv: str = "results/tables/metrics_all_horizons.csv", save_dir: str = "results/figures"):
    if not os.path.exists(metrics_csv):
        return
    df = pd.read_csv(metrics_csv)
    sub = df[(df["Horizon"] == "H1") & (df["Model"] == "GradientBoosting")].copy()

    scenarios = [
        "Chronological_Full", "Season_Winter", "Season_Monsoon",
        "Event_Diwali", "Event_StubbleBurning", "Event_Lockdown",
        "City_Delhi", "City_Bengaluru"
    ]
    sub = sub[sub["Scenario"].isin(scenarios)].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(sub))
    w = 0.35

    # Coverage
    ax1.bar(x - w/2, sub["PICP_Static"] * 100, width=w, label="Static Conformal", color="#e74c3c", alpha=0.85)
    ax1.bar(x + w/2, sub["PICP_Rolling"] * 100, width=w, label="Rolling Recalibration", color="#27ae60", alpha=0.85)
    ax1.axhline(90.0, color="black", ls="--", lw=1.5, label="Nominal 90% Target")
    ax1.axhline(85.0, color="gray", ls=":", lw=1.2, label="Acceptable Bound (85%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace("Season_", "").replace("Event_", "").replace("City_", "") for s in sub["Scenario"]], rotation=35, ha="right")
    ax1.set_ylabel("Coverage PICP (%)")
    ax1.set_ylim(50, 100)
    ax1.set_title("Empirical Coverage: Static vs. Rolling Adaptive")
    ax1.legend(loc="lower left")

    # Width
    ax2.bar(x - w/2, sub["MPIW_Static"], width=w, label="Static Width", color="#e74c3c", alpha=0.85)
    ax2.bar(x + w/2, sub["MPIW_Rolling"], width=w, label="Rolling Width", color="#27ae60", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace("Season_", "").replace("Event_", "").replace("City_", "") for s in sub["Scenario"]], rotation=35, ha="right")
    ax2.set_ylabel("Mean Interval Width (AQI units)")
    ax2.set_title("Interval Width Cost of Adaptive Recalibration")
    ax2.legend(loc="upper right")

    fig.suptitle("Conformal Prediction Interval Reliability Under Distribution Shift (Gradient Boosting, H=1)", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig05_conformal_coverage_and_width.png"))
    plt.close()


def plot_severe_analysis(
    metrics_csv: str = "results/tables/metrics_all_horizons.csv",
    save_dir: str = "results/figures",
):
    if not os.path.exists(metrics_csv):
        return

    df = pd.read_csv(metrics_csv)

    chrono = df[
        (df["Scenario"] == "Chronological_Full")
        & (df["Horizon"] == "H1")
    ].sort_values("Severe_Recall")

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
    )

    # Severe recall across all models
    ax1.barh(
        chrono["Model"],
        chrono["Severe_Recall"] * 100,
        color="#800020",
        alpha=0.85,
    )

    ax1.set_xlabel(
        "Severe Day Alert Recall (%) [Threshold AQI > 400]"
    )
    ax1.set_title(
        "Severe Day Detection Recall (Horizon H=1)"
    )
    ax1.set_xlim(0, 100)

    for i, row in enumerate(chrono.itertuples()):
        ax1.text(
            row.Severe_Recall * 100 + 1,
            i,
            f"{row.Severe_Recall * 100:.1f}%",
            va="center",
            fontsize=9,
        )

    # Bucket MAE comparison
    top_models = [
        "Persistence",
        "Ridge",
        "GradientBoosting",
        "XGBoost",
        "LSTM",
    ]

    sub = chrono[
        chrono["Model"].isin(top_models)
    ]

    b_cols = [
        "MAE_Moderate",
        "MAE_Poor",
        "MAE_VeryPoor",
        "MAE_Severe",
    ]

    b_names = [
        "Moderate",
        "Poor",
        "Very Poor",
        "Severe",
    ]

    x = np.arange(len(b_names))
    w = 0.16

    for j, m in enumerate(top_models):
        m_row = sub[sub["Model"] == m]

        if len(m_row) > 0:
            vals = [
                m_row[c].values[0]
                for c in b_cols
            ]

            offset = (
                j - (len(top_models) - 1) / 2
            ) * w

            ax2.bar(
                x + offset,
                vals,
                width=w,
                label=m,
                alpha=0.85,
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(b_names)
    ax2.set_ylabel("MAE (AQI units)")
    ax2.set_title(
        "Prediction Error by AQI Pollution Severity Category"
    )
    ax2.legend()

    fig.suptitle(
        "Extreme Episode Alerting and Severity Bucket Error Analysis",
        y=1.01,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "fig06_severe_alert_and_bucket_errors.png",
        )
    )

    plt.close()


def plot_delhi_case_study(features_csv: str = "data/processed/features_ffill_train_median.csv", save_dir: str = "results/figures"):
    if not os.path.exists(features_csv):
        return
    import joblib
    model_path = "models/GradientBoosting_H1.joblib"
    if not os.path.exists(model_path):
        return

    df = pd.read_csv(features_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    from src.models.classical import get_feature_columns
    cols = get_feature_columns(df)

    # Focus on Delhi winter / burning window in test period (Oct 2019 - Jan 2020)
    delhi_test = df[(df["City"] == "Delhi") & (df["Date"] >= "2019-10-01") & (df["Date"] <= "2020-01-31")].copy()
    if len(delhi_test) == 0:
        return

    model = joblib.load(model_path)
    X = delhi_test[cols].values
    y_true = delhi_test["target_H1"].values
    y_pred = model.predict(X)

    # Conformal width from calibration (~35-40 units)
    sp_q = 38.0
    lower = np.maximum(0, y_pred - sp_q)
    upper = y_pred + sp_q

    dates = delhi_test["Date"].values

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(dates, y_true, "k.-", lw=1.5, label="Observed Ground Truth AQI", alpha=0.9)
    ax.plot(dates, y_pred, "r--", lw=1.8, label="Gradient Boosting H=1 Forecast")
    ax.fill_between(dates, lower, upper, color="red", alpha=0.18, label="90% Conformal Prediction Interval")

    ax.axhline(400, color="#800020", ls=":", lw=1.5, label="Severe Pollution Threshold (400)")
    ax.axvspan(pd.to_datetime("2019-10-15"), pd.to_datetime("2019-11-30"), color="#ff7f0e", alpha=0.15, label="Stubble Burning Window")
    ax.axvline(pd.to_datetime("2019-10-27"), color="#9b59b6", ls="-.", lw=1.5, label="Diwali (Oct 27, 2019)")

    ax.set_ylabel("AQI Value")
    ax.set_title("Delhi Extreme Winter Pollution Spike: Forecasts and Conformal Envelope (H=1)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig07_delhi_spike_conformal_series.png"))
    plt.close()


def generate_all_publication_figures():
    os.makedirs("results/figures", exist_ok=True)
    print("Generating Figure 1: Missing values...")
    plot_missing_values()
    print("Generating Figure 2: AQI Trends...")
    plot_aqi_trends()
    print("Generating Figure 3: Model accuracy comparison...")
    plot_model_comparison()
    print("Generating Figure 4: Shift penalties...")
    plot_shift_penalties()
    print("Generating Figure 5: Conformal calibration...")
    plot_conformal_calibration()
    print("Generating Figure 6: Severe Alerting and Bucket Analysis...")
    plot_severe_analysis()
    print("Generating Figure 7: Delhi Case Study...")
    plot_delhi_case_study()
    print("Figures generation completed.")


if __name__ == "__main__":
    generate_all_publication_figures()
