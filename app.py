import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Afnan Barind Resilience Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA = Path(__file__).parent

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,0.06); padding: 12px 14px; border-radius: 12px;}
.small-note {font-size: 0.86rem; opacity: 0.78;}
.framework-box {padding: 12px 14px; border: 1px solid rgba(120,120,120,.25); border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    panel = pd.read_csv(DATA / "v02_monthly_panel.csv")
    panel["date_dt"] = pd.to_datetime(panel["date"] + "-01")
    bmda = pd.read_csv(DATA / "v01_bmda_t_panel.csv")
    warpo = pd.read_csv(DATA / "v01_warpo_baseline.csv")
    score = pd.read_csv(DATA / "v01_scorecard.csv")
    bbs = pd.read_csv(DATA / "v01_bbs_productivity_context.csv")
    climref = pd.read_csv(DATA / "v01_monthly_climate_reference.csv")
    interp = pd.read_csv(DATA / "modis_interpolation_audit.csv")
    qa = pd.read_csv(DATA / "modis_qa_comparison_audit.csv")
    loc = pd.read_csv(DATA / "upazila_locations.csv")
    return panel, bmda, warpo, score, bbs, climref, interp, qa, loc

panel, bmda, warpo, score, bbs, climref, interp, qa, loc = load_data()

# Month-specific anomalies remove the normal seasonal cycle before comparing shocks.
panel["rain_clim_mean"] = panel.groupby(["upazila","month"])["rainfall_mm"].transform("mean")
panel["rain_clim_std"] = panel.groupby(["upazila","month"])["rainfall_mm"].transform("std").replace(0, np.nan)
panel["rain_z"] = (panel["rainfall_mm"] - panel["rain_clim_mean"]) / panel["rain_clim_std"]
panel["ndvi_clim_mean"] = panel.groupby(["upazila","month"])["NDVI_monthly"].transform("mean")
panel["ndvi_clim_std"] = panel.groupby(["upazila","month"])["NDVI_monthly"].transform("std").replace(0, np.nan)
panel["ndvi_z"] = (panel["NDVI_monthly"] - panel["ndvi_clim_mean"]) / panel["ndvi_clim_std"]
panel["evi_clim_mean"] = panel.groupby(["upazila","month"])["EVI_monthly"].transform("mean")
panel["evi_clim_std"] = panel.groupby(["upazila","month"])["EVI_monthly"].transform("std").replace(0, np.nan)
panel["evi_z"] = (panel["EVI_monthly"] - panel["evi_clim_mean"]) / panel["evi_clim_std"]

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
SEASONS = {
    "All months": list(range(1,13)),
    "Dry / Rabi-Boro window (Nov-Apr)": [11,12,1,2,3,4],
    "Pre-monsoon (May)": [5],
    "Monsoon / Aman window (Jun-Oct)": [6,7,8,9,10],
}

# Sidebar filters
st.sidebar.title("Dashboard controls")
selected_upazilas = st.sidebar.multiselect(
    "Upazila", sorted(panel["upazila"].unique()), default=sorted(panel["upazila"].unique())
)
years = st.sidebar.slider("Year range", 2001, 2025, (2001, 2025))
season_name = st.sidebar.selectbox("Calendar / broad crop-season window", list(SEASONS))
months = SEASONS[season_name]
st.sidebar.caption("Season labels are broad analytical windows, not crop-specific phenology dates.")

f = panel[
    panel["upazila"].isin(selected_upazilas)
    & panel["year"].between(years[0], years[1])
    & panel["month"].isin(months)
].copy()

st.title("Afnan Barind Agricultural Technology & Resilience Dashboard")
st.caption("Mahadebpur and Sapahar, Naogaon • V0.1 technology/water baseline + V0.2 CHIRPS/MODIS longitudinal panel")

st.info(
    "This is an observational pilot dashboard. NDVI/EVI are vegetation-greenness proxies, not direct crop-yield measures. "
    "The derived groundwater stress score is an illustrative ordinal transformation, not the official IWM WSI. "
    "Final welfare U is not estimated."
)

# Framework strip
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown("<div class='framework-box'><b>T — Technology</b><br>BMDA DTW/LLP units, reported service area, coverage diagnostics.</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='framework-box'><b>P — Productivity proxy</b><br>MODIS NDVI/EVI, with BBS district crop data as context only.</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='framework-box'><b>R — Resilience</b><br>Vegetation response under rainfall shocks + WARPO water context.</div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='framework-box'><b>E — Resource pressure</b><br>Groundwater depth, stress-category exposure and water demand.</div>", unsafe_allow_html=True)

st.divider()

tabs = st.tabs(["Overview", "Climate & Vegetation", "Technology & Water Stress", "Resilience Explorer", "Quality & Method", "Downloads"])

# ---------------- OVERVIEW ----------------
with tabs[0]:
    st.subheader("Focal comparison")
    latest = score.copy()
    latest["dtw_coverage_pct"] = latest["dtw_coverage"] * 100
    latest["very_high_stress_pct"] = latest["very_high_stress_union_share"] * 100

    metric_cols = st.columns(4)
    metric_cols[0].metric("Monthly V0.2 observations", f"{len(panel):,}")
    metric_cols[1].metric("Period", "2001–2025")
    metric_cols[2].metric("Interpolated months", int((panel["MODIS_source_status"]=="interpolated").sum()))
    metric_cols[3].metric("Upazilas", panel["upazila"].nunique())

    left, right = st.columns([1.2,1])
    with left:
        loc2 = loc[loc["upazila"].isin(selected_upazilas)].copy()
        st.map(loc2, latitude="lat", longitude="lon", size=9000, zoom=8)
        st.caption("Map points use audited ADM3 centroids from the V0.1 spatial check.")
    with right:
        show_cols = ["upazila","latest_dtw_units","latest_dtw_area_ha","dtw_coverage_pct","ha_per_dtw","gwt_max_depth_m","very_high_stress_pct","stress_score_0_100"]
        st.dataframe(
            latest[show_cols].rename(columns={
                "upazila":"Upazila","latest_dtw_units":"Latest DTWs","latest_dtw_area_ha":"Latest DTW area (ha)",
                "dtw_coverage_pct":"DTW area / irrigable baseline (%)","ha_per_dtw":"Ha per DTW",
                "gwt_max_depth_m":"GWT max depth (m)","very_high_stress_pct":"Very-high-stress union share (%)",
                "stress_score_0_100":"Derived stress score (0–100)"
            }),
            hide_index=True, use_container_width=True
        )

    annual = f.groupby(["upazila","year"], as_index=False).agg(
        rainfall_mm=("rainfall_mm","sum"),
        mean_ndvi=("NDVI_monthly","mean"),
        mean_evi=("EVI_monthly","mean"),
    )
    c1,c2 = st.columns(2)
    with c1:
        fig = px.line(annual, x="year", y="rainfall_mm", color="upazila", markers=True,
                      labels={"rainfall_mm":"Rainfall (mm)","year":"Year","upazila":"Upazila"},
                      title="Rainfall across selected months")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(annual, x="year", y="mean_ndvi", color="upazila", markers=True,
                      labels={"mean_ndvi":"Mean NDVI","year":"Year","upazila":"Upazila"},
                      title="Mean NDVI across selected months")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- CLIMATE & VEGETATION ----------------
with tabs[1]:
    st.subheader("Longitudinal climate and vegetation")
    c1,c2 = st.columns(2)
    with c1:
        fig = px.line(f, x="date_dt", y="rainfall_mm", color="upazila",
                      labels={"date_dt":"Date","rainfall_mm":"Monthly rainfall (mm)","upazila":"Upazila"},
                      title="Monthly rainfall")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(f, x="date_dt", y="NDVI_monthly", color="upazila",
                      labels={"date_dt":"Date","NDVI_monthly":"Monthly NDVI","upazila":"Upazila"},
                      title="Monthly NDVI")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Calendar-month climatology")
    clim = f.groupby(["upazila","month"], as_index=False).agg(
        rainfall_mm=("rainfall_mm","mean"), NDVI=("NDVI_monthly","mean"), EVI=("EVI_monthly","mean")
    )
    clim["month_name"] = clim["month"].map(MONTH_NAMES)
    order = [MONTH_NAMES[m] for m in months]
    clim["month_name"] = pd.Categorical(clim["month_name"], categories=order, ordered=True)
    clim = clim.sort_values("month_name")
    c1,c2 = st.columns(2)
    with c1:
        fig = px.bar(clim, x="month_name", y="rainfall_mm", color="upazila", barmode="group",
                     labels={"month_name":"Month","rainfall_mm":"Mean rainfall (mm)","upazila":"Upazila"},
                     title="Mean monthly rainfall")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(clim, x="month_name", y="NDVI", color="upazila", markers=True,
                      labels={"month_name":"Month","NDVI":"Mean NDVI","upazila":"Upazila"},
                      title="Mean monthly NDVI")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Rainfall anomaly vs vegetation anomaly")
    scatter = f.dropna(subset=["rain_z","ndvi_z"]).copy()
    fig = px.scatter(scatter, x="rain_z", y="ndvi_z", color="upazila", opacity=0.65,
                     hover_data=["date","rainfall_mm","NDVI_monthly","MODIS_coverage_proxy"],
                     labels={"rain_z":"Rainfall anomaly (calendar-month z-score)","ndvi_z":"NDVI anomaly (calendar-month z-score)","upazila":"Upazila"},
                     title="Monthly anomaly association (descriptive, not causal)")
    fig.add_vline(x=0, line_dash="dot")
    fig.add_hline(y=0, line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)
    corr = scatter.groupby("upazila").apply(lambda x: x["rain_z"].corr(x["ndvi_z"]), include_groups=False).rename("Pearson r").reset_index()
    st.dataframe(corr, hide_index=True, use_container_width=False)

# ---------------- TECHNOLOGY & WATER ----------------
with tabs[2]:
    st.subheader("Technology (T) and resource pressure (E)")
    c1,c2 = st.columns(2)
    with c1:
        fig = px.line(bmda, x="fy", y="dtw_irrigated_area_ha", color="upazila", markers=True,
                      labels={"fy":"BMDA fiscal year","dtw_irrigated_area_ha":"Reported DTW service area (ha)","upazila":"Upazila"},
                      title="BMDA reported DTW service area")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Reported equipment service areas are not necessarily unique irrigated land; overlap is possible.")
    with c2:
        fig = px.line(bmda, x="fy", y="dtw_coverage_of_irrigable_area", color="upazila", markers=True,
                      labels={"fy":"BMDA fiscal year","dtw_coverage_of_irrigable_area":"DTW area / WARPO irrigable baseline","upazila":"Upazila"},
                      title="DTW service-area ratio to fixed WARPO irrigable baseline")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### WARPO/IWM focal baseline")
    warpo_display_cols = [
        "upazila","irrigable_area_ha","agricultural_demand_mcm_per_y","water_use_2021_mcm",
        "demand_2030_mcm","demand_2041_mcm","gwt_min_depth_m","gwt_max_depth_m","pra_scarcity",
        "very_high_stress_unions","high","moderate","low","very_low","very-high_share","derived_ordinal_stress_score_0-100"
    ]
    warpo_display_cols = [c for c in warpo_display_cols if c in warpo.columns]
    st.dataframe(warpo[warpo_display_cols], hide_index=True, use_container_width=True)

    stress_cols = [c for c in ["very_high_stress_unions","high","moderate","low","very_low"] if c in warpo.columns]
    if stress_cols:
        stress_long = warpo[["upazila"] + stress_cols].melt(id_vars="upazila", var_name="stress_class", value_name="union_count")
        fig = px.bar(stress_long, x="upazila", y="union_count", color="stress_class", barmode="stack",
                     labels={"upazila":"Upazila","union_count":"Number of unions","stress_class":"Stress class"},
                     title="Official IWM stress-category counts used in V0.1")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### BBS productivity context (Naogaon district only)")
    st.warning("These crop statistics are district-level context. They are not Mahadebpur- or Sapahar-specific P observations.")
    st.dataframe(bbs, hide_index=True, use_container_width=True)

# ---------------- RESILIENCE ----------------
with tabs[3]:
    st.subheader("Exploratory resilience under rainfall shocks")
    st.caption("This tab is diagnostic. It measures associations in vegetation anomalies during unusually dry months; it does not identify causal irrigation effects.")
    threshold = st.slider("Dry-shock threshold (rainfall z-score ≤)", -2.0, -0.5, -1.0, 0.1)
    rf = f.dropna(subset=["rain_z","ndvi_z","evi_z"]).copy()
    rf["dry_shock"] = rf["rain_z"] <= threshold
    rf["shock_label"] = np.where(rf["dry_shock"], "Dry-shock month", "Other month")

    summary = rf.groupby("upazila", as_index=False).agg(
        observations=("date","count"),
        dry_shock_months=("dry_shock","sum"),
        mean_ndvi_anomaly_all=("ndvi_z","mean"),
    )
    shock_stats = rf[rf["dry_shock"]].groupby("upazila", as_index=False).agg(
        mean_ndvi_anomaly_in_shocks=("ndvi_z","mean"),
        median_ndvi_anomaly_in_shocks=("ndvi_z","median"),
        mean_evi_anomaly_in_shocks=("evi_z","mean"),
        mean_rainfall_z_in_shocks=("rain_z","mean"),
    )
    summary = summary.merge(shock_stats, on="upazila", how="left")
    st.dataframe(summary.round(3), hide_index=True, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        comp = rf.groupby(["upazila","shock_label"], as_index=False)["ndvi_z"].mean()
        fig = px.bar(comp, x="upazila", y="ndvi_z", color="shock_label", barmode="group",
                     labels={"upazila":"Upazila","ndvi_z":"Mean NDVI anomaly","shock_label":"Month type"},
                     title="Vegetation anomaly in dry-shock vs other months")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(rf, x="rain_z", y="ndvi_z", color="upazila", symbol="dry_shock", opacity=.7,
                         hover_data=["date","rainfall_mm","NDVI_monthly","MODIS_total_valid_pixels"],
                         labels={"rain_z":"Rainfall z-score","ndvi_z":"NDVI z-score","upazila":"Upazila","dry_shock":"Dry shock"},
                         title="Shock-month diagnostic")
        fig.add_vline(x=threshold, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Technology × shock: what is and is not supported now")
    st.write(
        "BMDA technology observations are available only for a small set of fiscal years (2017–18 to 2020–21 in the V0.1 focal panel). "
        "The dashboard therefore does not present a causal Technology × Climate Shock coefficient as a settled result. "
        "The longitudinal climate/vegetation diagnostics are ready; a defensible interaction model needs careful annual/seasonal alignment and sensitivity analysis."
    )

# ---------------- QUALITY ----------------
with tabs[4]:
    st.subheader("Data quality, interpolation and assumptions")
    q1,q2,q3 = st.columns(3)
    q1.metric("Final monthly rows", len(panel))
    q2.metric("Interpolated MODIS months", int((panel["MODIS_source_status"]=="interpolated").sum()))
    q3.metric("Mean MODIS coverage proxy", f"{panel['MODIS_coverage_proxy'].mean():.1%}")

    st.markdown("#### Interpolated months")
    st.dataframe(interp, hide_index=True, use_container_width=True)
    st.markdown("#### Strict QA0 vs QA01 audit")
    st.dataframe(qa, hide_index=True, use_container_width=True)

    st.markdown("#### Method notes")
    st.markdown("""
- CHIRPS v3 pentad precipitation was aggregated to monthly polygon means for Mahadebpur and Sapahar.
- MODIS MOD13Q1 NDVI/EVI was scaled by 0.0001.
- The primary series accepts SummaryQA 0 and 1; strict SummaryQA 0 is retained for sensitivity/reference.
- Monthly NDVI/EVI is a valid-pixel-weighted mean of available 16-day composites.
- Seven fully missing upazila-months were linearly interpolated and remain explicitly flagged.
- `MODIS_coverage_proxy` is a relative diagnostic, not an exact land-area coverage percentage.
- V0.1 BMDA area / WARPO irrigable-area ratios use non-contemporaneous numerator and denominator years and are diagnostic only.
- Final welfare `U` is not estimated in this dashboard.
""")

# ---------------- DOWNLOADS ----------------
with tabs[5]:
    st.subheader("Download dashboard data")
    st.write("These are the clean app-level data files used by this dashboard.")
    downloads = [
        ("V0.2 monthly modeling panel", DATA/"v02_monthly_panel.csv"),
        ("V0.1 BMDA technology panel", DATA/"v01_bmda_t_panel.csv"),
        ("V0.1 WARPO baseline", DATA/"v01_warpo_baseline.csv"),
        ("V0.1 scorecard", DATA/"v01_scorecard.csv"),
        ("V0.1 BBS productivity context", DATA/"v01_bbs_productivity_context.csv"),
        ("MODIS interpolation audit", DATA/"modis_interpolation_audit.csv"),
        ("MODIS QA comparison audit", DATA/"modis_qa_comparison_audit.csv"),
    ]
    for label, path in downloads:
        st.download_button(label, path.read_bytes(), file_name=path.name, mime="text/csv", use_container_width=True)

st.divider()
st.caption("Afnan Barind Model pilot • T/P/R/E framework • Dashboard generated from the audited V0.1 workbook and finalized V0.2 CHIRPS/MODIS package.")
