# ============================================================
# STEP 7: STREAMLIT DASHBOARD
# ============================================================
# HOW TO RUN:
#   pip install streamlit plotly
#   streamlit run notebooks/STEP7_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="Smart City Air Quality Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F0F2F6; }
    .stMetric label { font-size: 14px !important; }
    h1 { color: #2C3E50; }
    h2 { color: #2980B9; }
</style>
""", unsafe_allow_html=True)


# ── LOAD DATA ──────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load all project data — cached so it only loads once"""
    data = {}
    files = {
        'main':       'data/model_ready_data.csv',
        'sql_q1':     'outputs/sql_query1_station_pm25.csv',
        'sql_q2':     'outputs/sql_query2_monthly_trend.csv',
        'sql_q3':     'outputs/sql_query3_seasonal.csv',
        'models':     'outputs/model_comparison.csv',
        'shap':       'outputs/shap_feature_importance.csv',
        # FIX: correct filename from STEP6B output
        'tpe_zone':   'outputs/tpe_by_zone.csv',
        'tpe_period': 'outputs/tpe_by_period.csv',
        'policy':     'outputs/policy_simulations.csv',
        # cross_city may or may not exist depending on your steps
        'crosscity':  'outputs/cross_city_results.csv',
    }
    for key, path in files.items():
        try:
            data[key] = pd.read_csv(path)
            if 'Date' in data[key].columns:
                data[key]['Date'] = pd.to_datetime(data[key]['Date'])
        except Exception:
            data[key] = None
    return data

data = load_data()
df   = data.get('main')


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🌿 Smart City AQ")
st.sidebar.markdown("**Spatio-Temporal Analysis**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "📊 Temporal Analysis",
     "🗺️ Spatial Analysis",
     "🤖 ML Models",
     "🔮 Policy Simulations",
     "📋 About Project"]
)

if df is not None:
    st.sidebar.markdown("---")
    city_filter = st.sidebar.multiselect(
        "Filter by City",
        options=df['City'].unique().tolist(),
        default=df['City'].unique().tolist()
    )

    if 'Year' in df.columns:
        year_range = st.sidebar.slider(
            "Year Range",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max()))
        )
        filtered_df = df[
            (df['City'].isin(city_filter)) &
            (df['Year'] >= year_range[0]) &
            (df['Year'] <= year_range[1])
        ]
    else:
        filtered_df = df[df['City'].isin(city_filter)]
else:
    filtered_df = pd.DataFrame()


# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.title("🌿 Spatio-Temporal Analysis of Traffic & Air Pollution")
    st.subheader("Smart City Planning Dashboard | Delhi & Bengaluru")
    st.markdown("---")

    if df is not None and len(filtered_df) > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_aqi = filtered_df['AQI'].mean() if 'AQI' in filtered_df.columns else 0
            st.metric("Average AQI", f"{avg_aqi:.0f}")

        with col2:
            avg_pm25 = filtered_df['PM2.5'].mean()
            st.metric("Average PM2.5", f"{avg_pm25:.1f} µg/m³")

        with col3:
            n_stations = filtered_df['StationId'].nunique()
            st.metric("Monitoring Stations", n_stations)

        with col4:
            n_days = filtered_df['Date'].nunique() if 'Date' in filtered_df.columns else 0
            st.metric("Days of Data", f"{n_days:,}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📍 City Summary")
            agg_dict = {'PM2.5': 'mean', 'StationId': 'nunique'}
            if 'AQI' in filtered_df.columns:
                agg_dict['AQI'] = 'mean'
            city_stats = filtered_df.groupby('City').agg(agg_dict).reset_index().round(2)
            city_stats.columns = ['City', 'Avg PM2.5'] + (
                ['Avg AQI'] if 'AQI' in filtered_df.columns else []) + ['Stations']
            st.dataframe(city_stats, use_container_width=True)

        with col2:
            st.subheader("📊 AQI Category Distribution")
            if 'AQI_Category' in filtered_df.columns:
                cat_counts = filtered_df['AQI_Category'].value_counts().reset_index()
                cat_counts.columns = ['Category', 'Count']
                fig = px.pie(cat_counts, values='Count', names='Category',
                             color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback: bin PM2.5 into categories
                bins   = [0, 30, 60, 90, 120, float('inf')]
                labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor']
                filtered_df['PM25_Cat'] = pd.cut(
                    filtered_df['PM2.5'], bins=bins, labels=labels)
                cat_counts = filtered_df['PM25_Cat'].value_counts().reset_index()
                cat_counts.columns = ['Category', 'Count']
                fig = px.pie(cat_counts, values='Count', names='Category',
                             title='PM2.5 Category Distribution',
                             color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)

        if data.get('sql_q1') is not None:
            st.subheader("📈 Station-wise PM2.5 Summary (SQL Query 1)")
            st.dataframe(data['sql_q1'], use_container_width=True)
    else:
        st.warning("No data loaded. Run Steps 1–6 first to generate output files.")


# ============================================================
# PAGE: TEMPORAL ANALYSIS
# ============================================================
elif page == "📊 Temporal Analysis":
    st.title("📊 Temporal Analysis")
    st.markdown("How does pollution change over time?")
    st.markdown("---")

    if len(filtered_df) > 0 and 'Date' in filtered_df.columns:

        st.subheader("Monthly PM2.5 Trend by City")
        monthly = (filtered_df
                   .groupby(['City', filtered_df['Date'].dt.to_period('M')])['PM2.5']
                   .mean()
                   .reset_index())
        monthly['Date'] = monthly['Date'].astype(str)
        fig = px.line(monthly, x='Date', y='PM2.5', color='City',
                      title='Monthly Average PM2.5',
                      labels={'PM2.5': 'PM2.5 (µg/m³)'},
                      color_discrete_map={'Delhi': '#E74C3C', 'Bengaluru': '#2980B9'})
        fig.add_hline(y=60, line_dash="dash", line_color="orange",
                      annotation_text="NAAQS (60 µg/m³)")
        fig.add_hline(y=15, line_dash="dash", line_color="green",
                      annotation_text="WHO (15 µg/m³)")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Seasonal PM2.5 Distribution")
            season_order = ['Winter', 'Post-Monsoon', 'Spring', 'Monsoon']
            if 'Season' in filtered_df.columns:
                fig = px.box(filtered_df, x='Season', y='PM2.5',
                             color='City',
                             category_orders={'Season': season_order},
                             color_discrete_map={'Delhi': '#E74C3C',
                                                 'Bengaluru': '#2980B9'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Season column not found in data.")

        with col2:
            st.subheader("PM2.5 Monthly Heatmap (Delhi)")
            delhi_df = filtered_df[filtered_df['City'] == 'Delhi']
            if len(delhi_df) > 0 and 'Year' in delhi_df.columns and 'Month' in delhi_df.columns:
                pivot = delhi_df.pivot_table(
                    values='PM2.5', index='Year', columns='Month', aggfunc='mean')
                fig = px.imshow(pivot,
                                color_continuous_scale='RdYlGn_r',
                                labels={'color': 'PM2.5'},
                                title='Delhi: PM2.5 by Year and Month')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Delhi data or Year/Month columns not available.")

        # Peak vs Off-peak from STEP6B
        if data.get('tpe_period') is not None:
            st.markdown("---")
            st.subheader("⏰ Peak vs Off-Peak PM2.5 (from STEP6B)")
            tp = data['tpe_period']
            fig = px.bar(tp, x='Period', y='Baseline_PM25',
                         color='Period',
                         text='Baseline_PM25',
                         title='Peak vs Off-Peak Average PM2.5',
                         labels={'Baseline_PM25': 'Avg PM2.5 (µg/m³)'},
                         color_discrete_sequence=['#E74C3C', '#3498DB'])
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(tp, use_container_width=True)
    else:
        st.warning("No filtered data available. Adjust filters.")


# ============================================================
# PAGE: SPATIAL ANALYSIS
# ============================================================
elif page == "🗺️ Spatial Analysis":
    st.title("🗺️ Spatial Analysis")
    st.markdown("Where is pollution worst? Which areas are high-risk?")
    st.markdown("---")

    if len(filtered_df) > 0:
        st.subheader("Average PM2.5 by Station")
        station_avg = (filtered_df
                       .groupby(['StationId', 'City'])
                       .agg(Avg_PM25=('PM2.5', 'mean'))
                       .reset_index()
                       .sort_values('Avg_PM25', ascending=False))

        fig = px.bar(station_avg, x='StationId', y='Avg_PM25',
                     color='City',
                     title='Average PM2.5 by Monitoring Station',
                     labels={'StationId': 'Station', 'Avg_PM25': 'PM2.5 (µg/m³)'},
                     color_discrete_map={'Delhi': '#E74C3C', 'Bengaluru': '#2980B9'})
        fig.add_hline(y=60, line_dash="dash", line_color="red",
                      annotation_text="NAAQS standard (60)")
        st.plotly_chart(fig, use_container_width=True)

        # TPE zone sensitivity chart
        if data.get('tpe_zone') is not None:
            st.markdown("---")
            st.subheader("🎯 TPE Zone Sensitivity (from STEP6B)")
            tz = data['tpe_zone'].copy()
            tz['ShortStation'] = tz['Station'].str.split(',').str[0]
            color_map = {'High': '#E74C3C', 'Medium': '#F39C12', 'Low': '#27AE60'}
            fig = px.bar(tz.sort_values('TPE', ascending=True),
                         x='TPE', y='ShortStation',
                         orientation='h',
                         color='Sensitivity',
                         color_discrete_map=color_map,
                         title='Traffic-Pollution Elasticity by Zone',
                         labels={'TPE': 'TPE Score', 'ShortStation': 'Station'})
            fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                          annotation_text="High threshold")
            fig.add_vline(x=0.3, line_dash="dash", line_color="orange",
                          annotation_text="Medium threshold")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🗺️ Interactive Maps")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('outputs/map_delhi.html'):
                st.markdown("**Delhi Pollution Map**")
                with open('outputs/map_delhi.html', 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=400)
            else:
                st.info("Map not found. Run Step 5 to generate maps.")
        with col2:
            if os.path.exists('outputs/map_bengaluru.html'):
                st.markdown("**Bengaluru Pollution Map**")
                with open('outputs/map_bengaluru.html', 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=400)
            else:
                st.info("Map not found. Run Step 5 to generate maps.")
    else:
        st.warning("No data available.")


# ============================================================
# PAGE: ML MODELS
# ============================================================
elif page == "🤖 ML Models":
    st.title("🤖 Machine Learning Models")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Comparison")
        if data.get('models') is not None:
            models_df = data['models']
            st.dataframe(models_df.round(4), use_container_width=True)
            # FIX: handle both 'R2' and 'R²' column names
            r2_col = 'R2' if 'R2' in models_df.columns else (
                     'R²' if 'R²' in models_df.columns else None)
            if r2_col:
                best_model = models_df.loc[models_df[r2_col].idxmax(), 'Model']
                st.success(f"✅ Best Model: **{best_model}**")
        else:
            st.info("model_comparison.csv not found. Run Step 6.")

    with col2:
        if data.get('models') is not None:
            metrics_df = data['models']
            # Only plot columns that actually exist
            available = [m for m in ['RMSE', 'MAE', 'R2', 'R²']
                         if m in metrics_df.columns]
            if available:
                fig = make_subplots(rows=1, cols=len(available),
                                    subplot_titles=[f"{m} {'↓' if m != 'R2' else '↑'}"
                                                    for m in available])
                colors = ['#3498DB', '#2ECC71', '#E74C3C']
                for i, metric in enumerate(available):
                    fig.add_trace(
                        go.Bar(x=metrics_df['Model'], y=metrics_df[metric],
                               marker_color=colors[:len(available)],
                               showlegend=False),
                        row=1, col=i + 1
                    )
                fig.update_layout(title_text="Model Performance Metrics", height=350)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 SHAP Feature Importance (XGBoost)")
    if data.get('shap') is not None:
        shap_df = data['shap'].head(15)
        # Handle different possible column names
        val_col = 'Mean_SHAP' if 'Mean_SHAP' in shap_df.columns else shap_df.columns[1]
        feat_col = 'Feature' if 'Feature' in shap_df.columns else shap_df.columns[0]
        fig = px.bar(shap_df.sort_values(val_col),
                     x=val_col, y=feat_col,
                     orientation='h',
                     title='Top Features: Impact on PM2.5 Predictions',
                     labels={val_col: 'Mean |SHAP| value', feat_col: 'Feature'},
                     color=val_col,
                     color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
        st.info("Higher SHAP value = feature has MORE impact on predictions")
    else:
        st.info("shap_feature_importance.csv not found. Run Step 6.")

    if data.get('crosscity') is not None:
        st.subheader("🌐 Cross-City Transferability")
        st.dataframe(data['crosscity'], use_container_width=True)


# ============================================================
# PAGE: POLICY SIMULATIONS
# ============================================================
elif page == "🔮 Policy Simulations":
    st.title("🔮 Policy Simulations & TPE Analysis")
    st.markdown("What happens to air quality if we reduce traffic?")
    st.markdown("---")

    # ── TPE Summary metrics ───────────────────────────────
    # FIX: derive tpe_val safely from tpe_zone file
    tpe_val = 0.45  # safe default
    if data.get('tpe_zone') is not None:
        tpe_val = float(data['tpe_zone']['TPE'].mean())

    st.subheader("Traffic-Pollution Elasticity (TPE)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average TPE (all zones)", f"{tpe_val:.3f}")
    with col2:
        st.metric("Interpretation",
                  f"10% traffic cut → {tpe_val * 10:.1f}% PM2.5 drop")
    with col3:
        st.metric("Policy Value", "High" if tpe_val > 0.4 else "Medium")

    # Zone breakdown table
    if data.get('tpe_zone') is not None:
        st.markdown("---")
        st.subheader("📊 TPE by Zone")
        tz = data['tpe_zone']
        color_map = {'High': '#E74C3C', 'Medium': '#F39C12', 'Low': '#27AE60'}
        fig = px.bar(tz.sort_values('TPE', ascending=False),
                     x='Station', y='TPE',
                     color='Sensitivity',
                     color_discrete_map=color_map,
                     title='TPE by Monitoring Station',
                     labels={'TPE': 'Traffic-Pollution Elasticity'})
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                      annotation_text="High threshold (0.5)")
        fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                      annotation_text="Medium threshold (0.3)")
        st.plotly_chart(fig, use_container_width=True)

    # Policy simulation results from STEP6B
    if data.get('policy') is not None:
        st.markdown("---")
        st.subheader("📋 Policy Simulation Results (from STEP6B)")
        pol = data['policy']
        fig = px.bar(pol,
                     x='Scenario', y='Simulated_PM25',
                     color='Scenario',
                     text='Simulated_PM25',
                     title='Simulated PM2.5 Under Each Policy',
                     labels={'Simulated_PM25': 'PM2.5 (µg/m³)'})
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        if 'Baseline_PM25' in pol.columns:
            fig.add_hline(y=pol['Baseline_PM25'].iloc[0],
                          line_dash="dash", line_color="grey",
                          annotation_text=f"Baseline: {pol['Baseline_PM25'].iloc[0]:.1f}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pol, use_container_width=True)

    # Interactive simulator
    st.markdown("---")
    st.subheader("🎮 Interactive Simulation")
    traffic_reduction = st.slider("Traffic Reduction (%)", 0, 50, 20)
    wind_increase     = st.slider("Wind Speed Increase (%)", 0, 100, 0)

    if len(filtered_df) > 0:
        base_pm25       = filtered_df['PM2.5'].mean()
        pm25_reduction  = base_pm25 * (traffic_reduction / 100) * tpe_val
        wind_reduction  = base_pm25 * (wind_increase / 100) * 0.15
        total_reduction = pm25_reduction + wind_reduction
        new_pm25        = max(0, base_pm25 - total_reduction)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Avg PM2.5", f"{base_pm25:.1f} µg/m³")
        with col2:
            st.metric("Simulated PM2.5",
                      f"{new_pm25:.1f} µg/m³",
                      delta=f"-{total_reduction:.1f} µg/m³")
        with col3:
            pct = (total_reduction / base_pm25) * 100 if base_pm25 > 0 else 0
            st.metric("Improvement", f"{pct:.1f}%")

        st.subheader("Planning Recommendations")
        if traffic_reduction >= 20:
            st.success("✅ **Odd-Even Scheme**: Implement during winter months "
                       "(Nov–Jan) for Delhi and Post-Monsoon for Bengaluru")
        if traffic_reduction >= 30:
            st.success("✅ **Low Emission Zones**: Designate near "
                       "Anand Vihar, ITO (Delhi) and Silk Board (Bengaluru)")
        if wind_increase > 0:
            st.info("ℹ️ Wind conditions improve dispersion — "
                    "schedule outdoor activities on windy days")
        st.warning("⚠️ **Sensor Placement**: Priority — Anand Vihar, Punjabi Bagh "
                   "(Delhi); Peenya, Silk Board (Bengaluru)")


# ============================================================
# PAGE: ABOUT PROJECT
# ============================================================
elif page == "📋 About Project":
    st.title("📋 About This Project")
    st.markdown("""
    ## Spatio-Temporal Analysis of Traffic and Air Pollution for Smart City Planning

    **BTech Major Project** | Computer Science & Engineering

    ---

    ### 🎯 Objectives
    1. Build an integrated SQL data pipeline for pollution, traffic & weather
    2. Perform spatio-temporal analysis identifying pollution hotspots
    3. Develop ML models (Linear Regression, Random Forest, XGBoost) for AQI prediction
    4. Apply SHAP explainability to understand traffic-pollution relationships
    5. Compute Traffic-Pollution Elasticity (TPE) for policy simulation
    6. Evaluate cross-city transferability (Delhi ↔ Bengaluru)

    ---

    ### 🔬 Models Used

    | Model | Purpose | R² Target |
    |-------|---------|-----------|
    | Linear Regression | Baseline benchmark | 0.4–0.6 |
    | Random Forest | Non-linear patterns | 0.65–0.75 |
    | XGBoost + SHAP | Best performance + explainability | 0.7–0.85 |

    ---

    ### 🗃️ Data Sources
    - **Air Quality**: CPCB station-wise data via Kaggle (2015–2020)
    - **Traffic**: Bangalore Traffic Pulse (Kaggle)
    - **Weather**: Open-Meteo Historical API

    ---

    ### 🏙️ Cities Analysed
    - **Delhi**: Anand Vihar, IGI Airport, ITO, Punjabi Bagh, RK Puram, Rohini...
    - **Bengaluru**: BTM Layout, Hebbal, Jayanagar, Peenya, Silk Board...

    ---

    ### 🆕 Novel Contributions
    1. First integrated SQL + ML + map pipeline for Indian smart cities
    2. Traffic-Pollution Elasticity (TPE) metric — novel contribution
    3. Cross-city model transferability test
    4. Interactive decision-support dashboard for urban planners
    """)


# ── FOOTER ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>BTech Major Project | Smart City Air Quality Analysis | "
    "Data: CPCB (via Kaggle) + Open-Meteo</small></center>",
    unsafe_allow_html=True
)
