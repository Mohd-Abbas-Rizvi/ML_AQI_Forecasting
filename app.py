# Streamlit Research Dashboard
# AQI Forecasting Under Distribution Shift

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="AQI Forecasting Under Distribution Shift",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_all_metrics():
    path = "results/tables/metrics_all_horizons.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_data
def load_city_data():
    path = "data/processed/features_ffill_train_median.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    return pd.DataFrame()


@st.cache_data
def load_ablations():
    tables = {}

    names = [
        "ablation_imputation",
        "ablation_features",
        "ablation_conformal_window",
        "spatial_held_out_city",
        "beijing_shift_results",
    ]

    for name in names:
        path = f"results/tables/{name}.csv"

        if os.path.exists(path):
            tables[name] = pd.read_csv(path)

    return tables


def filter_scenario(df, scenario):
    df = df.copy()

    if scenario in [
        "Chronological_Full",
        "City_Delhi",
        "City_Bengaluru",
        "City_Chennai",
        "City_Hyderabad",
        "City_Lucknow",
    ]:
        return df

    if scenario == "Year_2019":
        return df[df["Date"].dt.year == 2019]

    if scenario == "Year_2020":
        return df[df["Date"].dt.year == 2020]

    if scenario == "Season_Winter":
        return df[df["Date"].dt.month.isin([12, 1, 2])]

    if scenario == "Season_Summer":
        return df[df["Date"].dt.month.isin([3, 4, 5])]

    if scenario == "Season_Monsoon":
        return df[df["Date"].dt.month.isin([6, 7, 8, 9])]

    if scenario == "Season_PostMonsoon":
        return df[df["Date"].dt.month.isin([10, 11])]

    event_col_map = {
        "Event_Diwali": "is_diwali_window",
        "Event_StubbleBurning": "is_stubble_burning",
        "Event_Lockdown": "is_lockdown",
        "Event_WinterInversion": "is_winter_inversion",
    }

    if scenario in event_col_map:
        col = event_col_map[scenario]

        if col in df.columns:
            return df[df[col] == 1]

    return df


metrics_df = load_all_metrics()
features_df = load_city_data()
ablation_tables = load_ablations()


# Header
st.markdown(
    '<div class="main-title">'
    "AQI Forecasting Under Distribution Shift"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Research Dashboard: Forecast accuracy and conformal calibration "
    "under temporal, seasonal, event-driven and spatial distribution shifts."
    "</div>",
    unsafe_allow_html=True,
)


# Sidebar
st.sidebar.header("Configuration & Slices")

horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [
        "H1 (Next-Day)",
        "H3 (3-Day Ahead)",
    ],
    index=0,
)

horizon_key = "H1" if horizon.startswith("H1") else "H3"


if not metrics_df.empty:
    available_models = sorted(
        metrics_df["Model"].dropna().unique()
    )
else:
    available_models = ["GradientBoosting"]


default_idx = (
    available_models.index("GradientBoosting")
    if "GradientBoosting" in available_models
    else 0
)

selected_model = st.sidebar.selectbox(
    "Model",
    available_models,
    index=default_idx,
)


available_cities = [
    "Delhi",
    "Bengaluru",
    "Chennai",
    "Hyderabad",
    "Lucknow",
]

selected_city = st.sidebar.selectbox(
    "City Focus",
    available_cities,
    index=0,
)


scenario_options = [
    "Chronological_Full (2019 to mid-2020)",
    "Random_Reference (Reference Benchmark)",
    "Year_2019",
    "Year_2020",
    "Season_Winter",
    "Season_Summer",
    "Season_Monsoon",
    "Season_PostMonsoon",
    "Event_Diwali",
    "Event_StubbleBurning",
    "Event_Lockdown",
    "Event_WinterInversion",
    f"City_{selected_city}",
]

selected_scenario_raw = st.sidebar.selectbox(
    "Shift Regime / Evaluation Slice",
    scenario_options,
    index=0,
)

selected_scenario = selected_scenario_raw.split(" ")[0]


st.sidebar.markdown("---")

st.sidebar.info(
    "**CPCB Dataset Protocol**\n"
    "- Train: 2015-2018\n"
    "- Validation: final training-period split\n"
    "- Test: 2019-mid 2020\n"
    "- Nominal Conformal Target: 90%"
)


# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Forecast & Conformal Intervals",
        "⚖️ Shift Penalty & Model Benchmark",
        "🎯 Conformal Calibration & Reliability",
        "🔬 Controlled Ablation Studies",
        "🌍 Secondary Study (UCI Beijing PM2.5)",
    ]
)


# Tab 1
with tab1:
    st.subheader(
        f"Observed vs. Forecasted AQI "
        f"({selected_city}, {horizon})"
    )

    if not metrics_df.empty:
        sub_metrics = metrics_df[
            (metrics_df["Horizon"] == horizon_key)
            & (metrics_df["Model"] == selected_model)
            & (metrics_df["Scenario"] == selected_scenario)
        ].copy()
    else:
        sub_metrics = pd.DataFrame()

    col1, col2, col3, col4, col5 = st.columns(5)

    if not sub_metrics.empty:
        row = sub_metrics.iloc[0]

        col1.metric(
            "MAE (AQI)",
            f"{row['MAE']:.2f}",
        )

        col2.metric(
            "RMSE (AQI)",
            f"{row['RMSE']:.2f}",
        )

        col3.metric(
            "Shift Penalty",
            f"{row['ShiftPenalty']:.2f}",
        )

        picp = row.get("PICP_Rolling", np.nan)
        gap = row.get("CoverageGap_Rolling", np.nan)
        recall = row.get("Severe_Recall", np.nan)

        col4.metric(
            "90% Coverage (PICP)",
            (
                f"{picp * 100:.1f}%"
                if pd.notna(picp)
                else "N/A"
            ),
            delta=(
                f"{gap * 100:.1f}%"
                if pd.notna(gap)
                else None
            ),
        )

        col5.metric(
            "Severe Alert Recall",
            (
                f"{recall * 100:.1f}%"
                if pd.notna(recall)
                else "N/A"
            ),
        )

    else:
        st.warning(
            "No metrics found for the selected combination."
        )


    if not features_df.empty:
        city_sub = features_df[
            features_df["City"] == selected_city
        ].copy()

        test_sub = city_sub[
            city_sub["Date"] >= "2019-01-01"
        ].copy()

        target_c = f"target_{horizon_key}"
        valid_c = f"target_{horizon_key}_valid"

        if (
            target_c not in test_sub.columns
            or valid_c not in test_sub.columns
        ):
            st.error(
                f"Required target columns for "
                f"{horizon_key} are missing."
            )

        else:
            test_sub = test_sub[
                test_sub[valid_c] == 1
            ].copy()

            if selected_scenario == "Random_Reference":
                st.info(
                    "Random_Reference is not a continuous "
                    "chronological time slice. KPI metrics above "
                    "come from the saved random-reference experiment, "
                    "so no time-series chart is shown for this scenario."
                )

            else:
                test_sub = filter_scenario(
                    test_sub,
                    selected_scenario,
                )

                if test_sub.empty:
                    st.info(
                        "No valid observations are available for "
                        "this city, horizon and scenario."
                    )

                else:
                    preds = None

                    # LSTM results are precomputed.
                    # Live inference is disabled because PyTorch
                    # sequence inference can crash Streamlit on macOS.
                    if selected_model == "LSTM":
                        st.info(
                            "LSTM metrics shown above are loaded from "
                            "the verified precomputed experiment results. "
                            "Live LSTM inference is disabled in this "
                            "Streamlit dashboard because PyTorch sequence "
                            "inference can cause a native process crash "
                            "on macOS. The trained checkpoints remain "
                            f"available as models/LSTM_{horizon_key}.pt "
                            "and can be reproduced using "
                            "experiments/run_lstm.py."
                        )

                    # Other models use joblib
                    else:
                        mpath = (
                            f"models/{selected_model}_"
                            f"{horizon_key}.joblib"
                        )

                        if not os.path.exists(mpath):
                            st.error(
                                "Model artifact not found: "
                                f"{mpath}"
                            )

                        else:
                            try:
                                from src.models.classical import (
                                    get_feature_columns,
                                )

                                model = joblib.load(
                                    mpath
                                )

                                if selected_model in [
                                    "Persistence",
                                    "SeasonalNaive",
                                ]:
                                    preds = model.predict(
                                        test_sub
                                    )

                                else:
                                    cols = (
                                        get_feature_columns(
                                            city_sub
                                        )
                                    )

                                    X_val = test_sub[
                                        cols
                                    ].values

                                    if (
                                        selected_model
                                        == "Quantile_GBM"
                                    ):
                                        _, preds, _ = (
                                            model.predict(
                                                X_val
                                            )
                                        )

                                    else:
                                        preds = model.predict(
                                            X_val
                                        )

                            except Exception as exc:
                                st.error(
                                    "Unable to evaluate "
                                    f"{selected_model}: {exc}"
                                )


                    if preds is not None:
                        preds = np.asarray(
                            preds
                        ).reshape(-1)

                        if len(preds) != len(test_sub):
                            st.error(
                                "Prediction length does not "
                                "match evaluation data length."
                            )

                        else:
                            test_sub[
                                "Predicted_AQI"
                            ] = preds

                            half_width = None

                            if not sub_metrics.empty:
                                mpiw = (
                                    sub_metrics.iloc[0].get(
                                        "MPIW_Rolling",
                                        np.nan,
                                    )
                                )

                                if pd.notna(mpiw):
                                    half_width = (
                                        float(mpiw) / 2.0
                                    )

                            if half_width is not None:
                                test_sub[
                                    "Lower_Bound"
                                ] = np.maximum(
                                    0,
                                    preds - half_width,
                                )

                                test_sub[
                                    "Upper_Bound"
                                ] = (
                                    preds + half_width
                                )

                            fig, ax = plt.subplots(
                                figsize=(12, 4.5)
                            )

                            ax.plot(
                                test_sub["Date"],
                                test_sub[target_c],
                                color="black",
                                lw=1.2,
                                label=(
                                    "Observed Ground Truth "
                                    f"({target_c})"
                                ),
                            )

                            ax.plot(
                                test_sub["Date"],
                                test_sub["Predicted_AQI"],
                                color="#1f77b4",
                                lw=1.4,
                                ls="--",
                                label=(
                                    "Predicted AQI "
                                    f"({selected_model})"
                                ),
                            )

                            if half_width is not None:
                                ax.fill_between(
                                    test_sub["Date"],
                                    test_sub[
                                        "Lower_Bound"
                                    ],
                                    test_sub[
                                        "Upper_Bound"
                                    ],
                                    color="#1f77b4",
                                    alpha=0.18,
                                    label=(
                                        "Illustrative 90% interval "
                                        "band using reported rolling "
                                        "mean width"
                                    ),
                                )

                            ax.axhline(
                                400,
                                color="#7e0023",
                                ls=":",
                                lw=1.2,
                                label=(
                                    "Severe Pollution "
                                    "Threshold (400)"
                                ),
                            )

                            ax.set_xlabel(
                                "Date"
                            )

                            ax.set_ylabel(
                                "AQI Value"
                            )

                            ax.set_title(
                                f"{selected_city}: "
                                f"{selected_scenario} "
                                f"({selected_model}, "
                                f"{horizon_key})"
                            )

                            ax.legend(
                                loc="upper right",
                                fontsize=9,
                            )

                            plt.tight_layout()

                            st.pyplot(
                                fig
                            )

                            plt.close()

                            if half_width is not None:
                                st.caption(
                                    "The shaded band uses the "
                                    "scenario-level rolling mean "
                                    "interval width (MPIW). "
                                    "It is a visualization aid, "
                                    "not the stored point-wise "
                                    "conformal interval for each date."
                                )

    else:
        st.info(
            "Processed city feature dataset not found."
        )


# Tab 2
with tab2:
    st.subheader(
        f"Shift Penalty & Model Benchmark "
        f"({horizon_key})"
    )

    st.write(
        "**Shift Penalty Definition**: "
        "$\\text{Shift Penalty} = "
        "\\frac{\\text{MAE}_{\\text{shifted}}}"
        "{\\text{MAE}_{\\text{random reference}}}$. "
        "A value greater than 1.0 indicates degradation "
        "relative to the random-reference benchmark."
    )

    if not metrics_df.empty:
        chrono_df = metrics_df[
            (metrics_df["Horizon"] == horizon_key)
            & (
                metrics_df["Scenario"]
                == "Chronological_Full"
            )
        ].sort_values(
            "MAE"
        )

        if not chrono_df.empty:
            display_cols = [
                "Model",
                "MAE",
                "RMSE",
                "R2",
                "ShiftPenalty",
                "PICP_Rolling",
                "Severe_Recall",
                "MAE_Severe",
            ]

            st.dataframe(
                chrono_df[
                    display_cols
                ].style.format(
                    {
                        "MAE": "{:.2f}",
                        "RMSE": "{:.2f}",
                        "R2": "{:.3f}",
                        "ShiftPenalty": "{:.2f}",
                        "PICP_Rolling": "{:.1%}",
                        "Severe_Recall": "{:.1%}",
                        "MAE_Severe": "{:.1f}",
                    }
                ),
                width="stretch",
            )

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            ax.bar(
                chrono_df["Model"],
                chrono_df["ShiftPenalty"],
                color="#2c3e50",
                alpha=0.85,
            )

            ax.axhline(
                1.0,
                color="#e74c3c",
                ls="--",
                lw=1.2,
                label="No Penalty (1.0)",
            )

            ax.set_ylabel(
                "Shift Penalty"
            )

            ax.set_title(
                "Model Degradation Under "
                f"Chronological Shift ({horizon_key})"
            )

            plt.xticks(
                rotation=30,
                ha="right",
            )

            ax.legend()

            plt.tight_layout()

            st.pyplot(
                fig
            )

            plt.close()

        else:
            st.info(
                "Chronological benchmark data not found."
            )


# Tab 3
with tab3:
    st.subheader(
        "Conformal Prediction Interval Reliability "
        "Under Distribution Shift"
    )

    st.write(
        "Evaluating whether nominal 90% prediction intervals "
        "maintain empirical coverage under seasonal, "
        "event-driven and spatial shifts."
    )

    if not metrics_df.empty:
        piv = metrics_df[
            (metrics_df["Horizon"] == horizon_key)
            & (
                metrics_df["Model"]
                == selected_model
            )
        ].copy()

        if not piv.empty:
            display_cols = [
                "Scenario",
                "SampleCount",
                "PICP_Static",
                "MPIW_Static",
                "PICP_Rolling",
                "MPIW_Rolling",
                "CoverageGap_Rolling",
            ]

            st.dataframe(
                piv[
                    display_cols
                ].style.format(
                    {
                        "PICP_Static": "{:.1%}",
                        "MPIW_Static": "{:.1f}",
                        "PICP_Rolling": "{:.1%}",
                        "MPIW_Rolling": "{:.1f}",
                        "CoverageGap_Rolling": "{:+.1%}",
                    }
                ),
                width="stretch",
            )

        else:
            st.info(
                "No conformal metrics found "
                "for the selected model."
            )

    fig05 = (
        "results/figures/"
        "fig05_conformal_coverage_and_width.png"
    )

    if os.path.exists(fig05):
        st.image(
            fig05,
            caption=(
                "Conformal Interval Coverage and "
                "Width Tradeoff Under Shift "
                "(Gradient Boosting H1 case study)"
            ),
        )


# Tab 4
with tab4:
    st.subheader(
        "Controlled Ablation Experiments"
    )

    col_a, col_b = st.columns(
        2
    )

    with col_a:
        st.markdown(
            "#### 1. Imputation Methods Comparison"
        )

        if (
            "ablation_imputation"
            in ablation_tables
        ):
            st.dataframe(
                ablation_tables[
                    "ablation_imputation"
                ].style.format(
                    {
                        "MAE": "{:.2f}",
                        "RMSE": "{:.2f}",
                        "R2": "{:.3f}",
                    }
                ),
                width="stretch",
            )

        st.markdown(
            "#### 3. Conformal Rolling Window Sensitivity"
        )

        if (
            "ablation_conformal_window"
            in ablation_tables
        ):
            st.dataframe(
                ablation_tables[
                    "ablation_conformal_window"
                ].style.format(
                    {
                        "PICP": "{:.1%}",
                        "MPIW": "{:.1f}",
                        "CoverageGap": "{:+.1%}",
                    }
                ),
                width="stretch",
            )


    with col_b:
        st.markdown(
            "#### 2. Feature Groupings "
            "(Lags vs. Pollutants)"
        )

        if (
            "ablation_features"
            in ablation_tables
        ):
            st.dataframe(
                ablation_tables[
                    "ablation_features"
                ].style.format(
                    {
                        "MAE": "{:.2f}",
                        "RMSE": "{:.2f}",
                        "R2": "{:.3f}",
                    }
                ),
                width="stretch",
            )

        st.markdown(
            "#### 4. Spatial Generalization "
            "(Held-out City Transfer)"
        )

        if (
            "spatial_held_out_city"
            in ablation_tables
        ):
            st.dataframe(
                ablation_tables[
                    "spatial_held_out_city"
                ].style.format(
                    {
                        "MAE": "{:.2f}",
                        "RMSE": "{:.2f}",
                        "R2": "{:.3f}",
                    }
                ),
                width="stretch",
            )


# Tab 5
with tab5:
    st.subheader(
        "Secondary Dataset: UCI Beijing PM2.5 "
        "Transfer Study (2010-2014)"
    )

    st.write(
        "The secondary benchmark uses 43,824 hourly "
        "observations aggregated to 1,826 daily records. "
        "The experiment uses 2010-2012 as the earlier "
        "training regime and 2013-2014 as the shifted "
        "evaluation period."
    )

    if (
        "beijing_shift_results"
        in ablation_tables
    ):
        st.dataframe(
            ablation_tables[
                "beijing_shift_results"
            ].style.format(
                {
                    "MAE_Test_2013_2014": "{:.2f}",
                    "RMSE_Test_2013_2014": "{:.2f}",
                    "R2_Test_2013_2014": "{:.3f}",
                    "ShiftPenalty": "{:.2f}",
                    "PICP_Static": "{:.1%}",
                    "MPIW_Static": "{:.1f}",
                    "PICP_Rolling": "{:.1%}",
                    "MPIW_Rolling": "{:.1f}",
                }
            ),
            width="stretch",
        )

    else:
        st.info(
            "Beijing shift results table not found."
        )