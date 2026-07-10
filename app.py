import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from databricks import sql as dbsql
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="iSpot Impression Analysis", page_icon="\U0001f4fa", layout="wide")

# ---- Config ----
def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)

db_host = get_secret("DATABRICKS_HOST")
db_token = get_secret("DATABRICKS_TOKEN")
db_warehouse = get_secret("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH")
llm_model = get_secret("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")

COLORS = {
    "navy": "#1B2A4A", "cyan": "#00BCD4", "light_cyan": "#80DEEA",
    "lime": "#C5E063", "white": "#FFFFFF", "light_gray": "#F8F9FA", "mid_gray": "#E0E0E0",
}

VIEW = "locality_dev.silver.ispot_dma_reports_latest"
MAPPING = "locality_dev.silver.freewheel_placement_mapping"


def get_llm_client():
    host = db_host.rstrip("/").replace("https://", "").replace("http://", "")
    return OpenAI(
        api_key=db_token,
        base_url=f"https://{host}/serving-endpoints",
    )


def run_query(sql):
    conn = dbsql.connect(
        server_hostname=db_host.replace("https://", "").replace("http://", ""),
        http_path=db_warehouse, access_token=db_token,
    )
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        cur.close()
        conn.close()


def format_number(n):
    if n is None or pd.isna(n):
        return "\u2014"
    n = float(n)
    if abs(n) >= 1e9: return f"{n/1e9:.1f}B"
    elif abs(n) >= 1e6: return f"{n/1e6:.1f}M"
    elif abs(n) >= 1e3: return f"{n/1e3:.0f}K"
    return f"{n:,.0f}"


def generate_insights(brand, campaign_label, metrics, dma_df, selected_dmas_count):
    """Use LLM to generate an executive summary of the campaign performance."""
    m = metrics
    pct = float(m.get("incrementality_pct", 0) or 0)
    ott_imp = float(m.get("ott_only_impressions", 0) or 0)
    ott_tv = float(m.get("ott_tv_impressions", 0) or 0)
    tv_only = float(m.get("tv_only_impressions", 0) or 0)
    ott_viewers = float(m.get("ott_total_viewers", 0) or 0)
    incr_viewers = float(m.get("ott_incremental_viewers", 0) or 0)
    ott_freq = float(m.get("ott_avg_frequency", 0) or 0)
    lin_freq = float(m.get("linear_avg_frequency", 0) or 0)

    # Top/bottom DMAs
    top_dmas = ""
    bottom_dmas = ""
    if dma_df is not None and not dma_df.empty and "incrementality_pct" in dma_df.columns:
        sorted_df = dma_df.sort_values("incrementality_pct", ascending=False)
        top_3 = sorted_df.head(3)
        bot_3 = sorted_df.tail(3)
        top_dmas = ", ".join(f"{r['dma']} ({r['incrementality_pct']:.0f}%)" for _, r in top_3.iterrows())
        bottom_dmas = ", ".join(f"{r['dma']} ({r['incrementality_pct']:.0f}%)" for _, r in bot_3.iterrows())

    prompt = f"""You are an advertising analytics expert at Locality, a streaming/OTT advertising company.
Write a concise 3-4 sentence executive summary for this campaign performance report.
Be specific with numbers. Highlight what's notable (good or concerning).

Brand: {brand}
Campaign: {campaign_label}
DMAs Analyzed: {selected_dmas_count}

KEY METRICS:
- Incrementality: {pct:.1f}% (OTT viewers NOT reached by linear TV)
- OTT Total Viewers: {format_number(ott_viewers)}
- OTT Incremental Viewers: {format_number(incr_viewers)}
- OTT Only Impressions: {format_number(ott_imp)}
- OTT + TV Overlap Impressions: {format_number(ott_tv)}
- TV Only Impressions: {format_number(tv_only)}
- OTT Average Frequency: {ott_freq:.1f}
- Linear Average Frequency: {lin_freq:.1f}

TOP PERFORMING DMAs: {top_dmas or 'N/A'}
LOWEST PERFORMING DMAs: {bottom_dmas or 'N/A'}

Write the summary now. Be direct, data-driven, and actionable. No bullet points — flowing prose only."""

    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a concise advertising analytics expert. Write executive summaries that highlight key takeaways and actionable insights."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"_Insights unavailable: {e}_"


# ---- Sidebar: Connection Status ----
with st.sidebar:
    st.title("\U0001f4fa iSpot Analysis")
    if all([db_host, db_token, db_warehouse]):
        st.success("Connected")
    else:
        st.error("Missing credentials in secrets/.env")
        st.stop()

# ---- Check connection ----
if not all([db_host, db_token, db_warehouse]):
    st.warning("Configure credentials.")
    st.stop()

# ---- Step 1: Brand Selection (Required) ----
st.title("\U0001f4fa iSpot Impression Analysis")
st.caption("Select filters to generate your campaign performance report.")

st.markdown("---")
st.subheader("1. Select Brand")

@st.cache_data(ttl=300)
def get_brands():
    df = run_query(f"""
        SELECT DISTINCT brand
        FROM {VIEW}
        WHERE brand IS NOT NULL AND TRIM(brand) != ''
        ORDER BY brand ASC
    """)
    return df["brand"].tolist()

brands = get_brands()
selected_brand = st.selectbox("Brand / Advertiser (required)", [""] + brands, index=0)

if not selected_brand:
    st.info("\u2191 Select a brand to continue.")
    st.stop()

# ---- Step 2: Campaign Selection (Optional) ----
st.markdown("---")
st.subheader("2. Select Campaign (optional)")

@st.cache_data(ttl=300)
def get_campaigns(brand):
    """Get campaigns with names from the mapping table."""
    df = run_query(f"""
        SELECT
            v.campaign_id,
            COALESCE(m.locality_campaign, CONCAT('Campaign ', v.campaign_id)) AS campaign_name,
            SUM(v.ott_total_impressions) AS ott_imp
        FROM {VIEW} v
        LEFT JOIN {MAPPING} m
            ON v.campaign_id = CAST(m.locality_campaign_id AS BIGINT)
        WHERE LOWER(v.brand) = LOWER(\'{brand.replace(chr(39), chr(39)+chr(39))}\')
        GROUP BY v.campaign_id, COALESCE(m.locality_campaign, CONCAT('Campaign ', v.campaign_id))
        ORDER BY ott_imp DESC
    """)
    return df

campaigns_df = get_campaigns(selected_brand)
# Build display labels: "Campaign Name (ID: 12345)"
campaign_labels = ["All Campaigns"] + [
    f"{row['campaign_name']} (ID: {row['campaign_id']})"
    for _, row in campaigns_df.iterrows()
]
selected_campaign_label = st.selectbox("Campaign", campaign_labels)

# Extract campaign_id from selection
selected_campaign_id = None
if selected_campaign_label != "All Campaigns":
    idx = campaign_labels.index(selected_campaign_label) - 1
    selected_campaign_id = int(campaigns_df.iloc[idx]["campaign_id"])

# Build base WHERE clause
brand_escaped = selected_brand.replace("'", "''")
where_clauses = [f"LOWER(brand) = LOWER(\'{brand_escaped}\')"]
if selected_campaign_id is not None:
    where_clauses.append(f"campaign_id = {selected_campaign_id}")

# ---- Step 3: DMA Selection ----
st.markdown("---")
st.subheader("3. DMA Breakout")

@st.cache_data(ttl=300)
def get_dmas(brand, campaign_id=None):
    where = f"LOWER(brand) = LOWER(\'{brand.replace(chr(39), chr(39)+chr(39))}\')"
    if campaign_id:
        where += f" AND campaign_id = {campaign_id}"
    df = run_query(f"""
        SELECT DISTINCT dma, SUM(ott_total_impressions) as ott_imp
        FROM {VIEW} WHERE {where}
        GROUP BY dma ORDER BY dma ASC
    """)
    return df["dma"].tolist()

available_dmas = get_dmas(selected_brand, selected_campaign_id)
dma_choice = st.radio("DMA scope", ["Full DMA Breakout (all DMAs)", "Select Specific DMAs"], horizontal=True)

selected_dmas = available_dmas
if dma_choice == "Select Specific DMAs":
    selected_dmas = st.multiselect(
        f"Choose DMAs ({len(available_dmas)} available)",
        available_dmas,
        default=available_dmas[:5] if len(available_dmas) > 5 else available_dmas,
    )
    if not selected_dmas:
        st.warning("Select at least one DMA.")
        st.stop()
    dma_list = ", ".join(f"\'{d.replace(chr(39), chr(39)+chr(39))}\'" for d in selected_dmas)
    where_clauses.append(f"dma IN ({dma_list})")

# ---- Step 4: Date Range ----
st.markdown("---")
st.subheader("4. Date Range")
date_choice = st.radio("Analysis period", ["Full Campaign (all available data)", "Custom Date Range"], horizontal=True)

if date_choice == "Custom Date Range":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    where_clauses.append(f"report_date >= \'{start_date}\'")
    where_clauses.append(f"report_date <= \'{end_date}\'")

# ---- Generate Report ----
st.markdown("---")
where_sql = " AND ".join(where_clauses)

if st.button("\U0001f4ca Generate Report", type="primary", use_container_width=True):
    with st.spinner("Querying data..."):
        # Main metrics query
        metrics_df = run_query(f"""
            SELECT
                SUM(ott_incremental_impressions) AS ott_only_impressions,
                CAST(SUM(ott_overlap_with_linear_overlap) AS BIGINT) AS ott_tv_impressions,
                CAST(SUM(linear_only_impressions) AS BIGINT) AS tv_only_impressions,
                SUM(ott_total_impressions) AS ott_total_impressions,
                CAST(SUM(linear_total_impressions) AS BIGINT) AS linear_total_impressions,
                SUM(total_impressions) AS total_impressions,
                SUM(ott_incremental_viewers) AS ott_incremental_viewers,
                SUM(ott_total_viewers) AS ott_total_viewers,
                CAST(SUM(overlap_viewers) AS BIGINT) AS overlap_viewers,
                SUM(all_viewers) AS all_viewers,
                ROUND(SUM(ott_incremental_viewers) * 100.0 / NULLIF(SUM(ott_total_viewers), 0), 1) AS incrementality_pct,
                ROUND(SUM(ott_total_impressions) * 1.0 / NULLIF(SUM(ott_total_viewers), 0), 1) AS ott_avg_frequency,
                ROUND(SUM(linear_total_impressions) * 1.0 / NULLIF(CAST(SUM(linear_viewers) AS BIGINT), 0), 1) AS linear_avg_frequency
            FROM {VIEW}
            WHERE {where_sql}
        """)

        # DMA breakdown
        dma_df = run_query(f"""
            SELECT dma,
                SUM(ott_total_impressions) AS ott_impressions,
                SUM(ott_total_viewers) AS ott_viewers,
                SUM(ott_incremental_viewers) AS incremental_viewers,
                ROUND(SUM(ott_incremental_viewers) * 100.0 / NULLIF(SUM(ott_total_viewers), 0), 1) AS incrementality_pct
            FROM {VIEW}
            WHERE {where_sql}
            GROUP BY dma
            ORDER BY ott_impressions DESC
        """)

    if metrics_df.empty or metrics_df.iloc[0]["ott_total_impressions"] is None:
        st.error("No data found for the selected filters.")
        st.stop()

    m = metrics_df.iloc[0]
    pct = float(m["incrementality_pct"] or 0)

    # ---- RENDER DASHBOARD ----
    campaign_display = selected_campaign_label if selected_campaign_label != "All Campaigns" else "All Campaigns"
    st.markdown(f"### {selected_brand} — {campaign_display}")

    # Incrementality Banner
    st.markdown(
        f'<div style="background:{COLORS["lime"]}; border-radius:8px; padding:20px 28px; margin:12px 0;">'
        f'<span style="color:{COLORS["navy"]}; font-size:20px; font-weight:700;">'
        f'{pct:.0f}% of your streaming campaign reached consumers not reached with linear TV ads.'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # KPI Cards + Donut
    col_donut, col_kpis = st.columns([1, 3])
    with col_donut:
        fig = go.Figure(go.Pie(
            values=[pct, 100 - pct], hole=0.7,
            marker_colors=[COLORS["navy"], COLORS["mid_gray"]],
            textinfo="none", hoverinfo="skip", sort=False))
        fig.add_annotation(text=f"<b>{pct:.0f}%</b>", x=0.5, y=0.5,
            font_size=28, font_color=COLORS["navy"], showarrow=False)
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
            height=200, width=200, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=False)

    with col_kpis:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Incrementality", f"{pct:.1f}%")
        k2.metric("OTT Viewers", format_number(m["ott_total_viewers"]))
        k3.metric("OTT Incremental", format_number(m["ott_incremental_viewers"]))
        k4.metric("All Viewers", format_number(m["all_viewers"]))

    # Media Buying Pie
    st.markdown("#### Media Buying Breakdown")
    ott_only = float(m["ott_only_impressions"] or 0)
    ott_tv = float(m["ott_tv_impressions"] or 0)
    tv_only = float(m["tv_only_impressions"] or 0)
    total_imp = ott_only + ott_tv + tv_only

    col_pie, col_metrics = st.columns([1, 2])
    with col_pie:
        fig = go.Figure(go.Pie(
            labels=["OTT Only", "OTT + TV", "TV Only"],
            values=[ott_only, ott_tv, tv_only],
            marker_colors=[COLORS["navy"], COLORS["cyan"], COLORS["light_cyan"]],
            textinfo="label+percent", textfont_size=12))
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=280,
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with col_metrics:
        st.metric("OTT Only Impressions", format_number(ott_only), f"{ott_only/total_imp*100:.1f}% of total" if total_imp else "")
        st.metric("OTT + TV Impressions", format_number(ott_tv), f"{ott_tv/total_imp*100:.1f}% of total" if total_imp else "")
        st.metric("TV Only Impressions", format_number(tv_only), f"{tv_only/total_imp*100:.1f}% of total" if total_imp else "")

    # Frequency Comparison
    st.markdown("#### Average Frequency")
    ott_freq = float(m["ott_avg_frequency"] or 0)
    lin_freq = float(m["linear_avg_frequency"] or 0)
    col_freq, col_vals = st.columns([2, 1])
    with col_freq:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Locality OTT"], y=[ott_freq], marker_color=COLORS["navy"],
            text=[f"{ott_freq:.1f}"], textposition="outside", showlegend=False))
        fig.add_trace(go.Bar(x=["TV Market"], y=[lin_freq], marker_color=COLORS["light_cyan"],
            text=[f"{lin_freq:.1f}"], textposition="outside", showlegend=False))
        fig.update_layout(yaxis=dict(range=[0, max(ott_freq, lin_freq, 1) * 1.4]),
            margin=dict(t=20, b=30, l=40, r=20), height=250,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLORS["light_gray"])
        st.plotly_chart(fig, use_container_width=True)
    with col_vals:
        st.metric("OTT Frequency", f"{ott_freq:.1f}")
        st.metric("Linear Frequency", f"{lin_freq:.1f}")

    # DMA Breakout
    st.markdown("#### DMA Breakout")
    if not dma_df.empty:
        display_dma = dma_df.head(20)
        fig = go.Figure(go.Bar(
            y=display_dma["dma"], x=display_dma["ott_impressions"], orientation="h",
            marker_color=COLORS["cyan"],
            text=display_dma["ott_impressions"].apply(format_number),
            textposition="outside"))
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=20, b=30, l=180, r=80),
            height=max(300, len(display_dma) * 30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLORS["light_gray"])
        st.plotly_chart(fig, use_container_width=True)

        with st.expander(f"Full DMA Data ({len(dma_df)} DMAs)"):
            st.dataframe(dma_df, use_container_width=True)

    # ---- LLM INSIGHTS NARRATIVE ----
    st.markdown("---")
    st.markdown("#### \U0001f4a1 Executive Insights")
    with st.spinner("Generating insights..."):
        insights = generate_insights(
            brand=selected_brand,
            campaign_label=campaign_display,
            metrics=m.to_dict(),
            dma_df=dma_df,
            selected_dmas_count=len(selected_dmas),
        )
    st.markdown(
        f'<div style="background:{COLORS["light_gray"]}; border-left:4px solid {COLORS["cyan"]}; '
        f'border-radius:4px; padding:16px 20px; margin:8px 0; font-size:15px; line-height:1.6;">'
        f'{insights}</div>',
        unsafe_allow_html=True,
    )
