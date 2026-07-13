import os
import re
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
llm_model = get_secret("LLM_MODEL", "databricks-claude-sonnet-4-6")

COLORS = {
    "navy": "#1B2A4A", "cyan": "#00BCD4", "light_cyan": "#80DEEA",
    "lime": "#C5E063", "white": "#FFFFFF", "light_gray": "#F8F9FA", "mid_gray": "#E0E0E0",
}

VIEW = "locality_dev.silver.ispot_dma_reports_latest"
MAPPING = "locality_dev.silver.freewheel_placement_mapping"


# ---- LLM Client ----
def get_llm_client():
    host = db_host.rstrip("/").replace("https://", "").replace("http://", "")
    return OpenAI(api_key=db_token, base_url=f"https://{host}/serving-endpoints")


# ---- SQL Runner ----
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


# ---- Helpers ----
def format_number(n):
    if n is None or pd.isna(n):
        return "\u2014"
    n = float(n)
    if abs(n) >= 1e9: return f"{n/1e9:.1f}B"
    elif abs(n) >= 1e6: return f"{n/1e6:.1f}M"
    elif abs(n) >= 1e3: return f"{n/1e3:.0f}K"
    return f"{n:,.0f}"


def generate_insights(brand, campaign_label, metrics, dma_df, dma_count, placement_info=None):
    """Use LLM to generate an executive summary of campaign performance."""
    m = metrics
    pct = float(m.get("incrementality_pct", 0) or 0)
    ott_imp = float(m.get("ott_only_impressions", 0) or 0)
    ott_tv = float(m.get("ott_tv_impressions", 0) or 0)
    tv_only = float(m.get("tv_only_impressions", 0) or 0)
    ott_viewers = float(m.get("ott_total_viewers", 0) or 0)
    incr_viewers = float(m.get("ott_incremental_viewers", 0) or 0)
    ott_freq = float(m.get("ott_avg_frequency", 0) or 0)
    lin_freq = float(m.get("linear_avg_frequency", 0) or 0)

    top_dmas = ""
    bottom_dmas = ""
    if dma_df is not None and not dma_df.empty and "incrementality_pct" in dma_df.columns:
        sorted_df = dma_df.sort_values("incrementality_pct", ascending=False)
        top_3 = sorted_df.head(3)
        bot_3 = sorted_df.tail(3)
        top_dmas = ", ".join(f"{r['dma']} ({r['incrementality_pct']:.0f}%)" for _, r in top_3.iterrows())
        bottom_dmas = ", ".join(f"{r['dma']} ({r['incrementality_pct']:.0f}%)" for _, r in bot_3.iterrows())

    placement_context = ""
    if placement_info:
        placement_context = f"""
Placement: {placement_info.get('locality_placement_name', 'N/A')}
Agency: {placement_info.get('locality_agency', 'N/A')}
Product: {placement_info.get('locality_product', 'N/A')}
Flight: {placement_info.get('locality_placement_start_date', '?')} to {placement_info.get('locality_placement_end_date', '?')}"""

    prompt = f"""You are an advertising analytics expert at Locality, a streaming/OTT advertising company.
Write a concise 3-4 sentence executive summary for this campaign performance report.
Be specific with numbers. Highlight what's notable (good or concerning).

Brand: {brand}
Campaign: {campaign_label}{placement_context}
DMAs Analyzed: {dma_count}

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

Write the summary now. Be direct, data-driven, and actionable. No bullet points - flowing prose only."""

    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a concise advertising analytics expert. Write executive summaries that highlight key takeaways and actionable insights."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400, temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"_Insights unavailable: {e}_"


# ---- Sidebar ----
with st.sidebar:
    st.title("\U0001f4fa iSpot Analysis")
    if all([db_host, db_token, db_warehouse]):
        st.success("Connected")
    else:
        st.error("Missing credentials in secrets/.env")
        st.stop()

if not all([db_host, db_token, db_warehouse]):
    st.warning("Configure credentials.")
    st.stop()


# ==============================================================
# STEP 1: BRAND SELECTION (Required)
# ==============================================================
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


# ==============================================================
# STEP 2: CAMPAIGN SELECTION (Optional)
# ==============================================================
st.markdown("---")
st.subheader("2. Select Campaign (optional)")

@st.cache_data(ttl=300)
def get_campaigns(brand):
    brand_esc = brand.replace("'", "''")
    df = run_query(f"""
        SELECT
            v.campaign_id,
            COALESCE(
                MAX(m.locality_advertiser),
                CONCAT('Campaign ', v.campaign_id)
            ) AS campaign_name,
            MIN(m.locality_campaign_start_date) AS start_date,
            MAX(m.locality_campaign_end_date) AS end_date,
            SUM(v.ott_total_impressions) AS ott_imp
        FROM {VIEW} v
        LEFT JOIN {MAPPING} m
            ON v.campaign_id = CAST(m.fw_campaign_id AS BIGINT)
        WHERE LOWER(v.brand) = LOWER('{brand_esc}')
        GROUP BY v.campaign_id
        ORDER BY ott_imp DESC
    """)
    return df

campaigns_df = get_campaigns(selected_brand)
def _fmt_date(d):
    """Convert ISO date string to MM/DD/YY format."""
    if not d or d == 'None':
        return None
    try:
        from datetime import datetime
        dt = datetime.strptime(str(d)[:10], "%Y-%m-%d")
        return dt.strftime("%m/%d/%y")
    except (ValueError, TypeError):
        return str(d)

campaign_labels = ["All Campaigns"] + [
    f"{row['campaign_name']} ({_fmt_date(row['start_date'])} - {_fmt_date(row['end_date'])})"
    if row.get('start_date') and row.get('end_date') and str(row['start_date']) != 'None'
    else f"{row['campaign_name']} (ID: {row['campaign_id']})"
    for _, row in campaigns_df.iterrows()
]
selected_campaign_label = st.selectbox("Campaign", campaign_labels)

selected_campaign_id = None
if selected_campaign_label != "All Campaigns":
    idx = campaign_labels.index(selected_campaign_label) - 1
    selected_campaign_id = int(campaigns_df.iloc[idx]["campaign_id"])

# Build WHERE clause
brand_escaped = selected_brand.replace("'", "''")
where_clauses = [f"LOWER(brand) = LOWER('{brand_escaped}')"]
if selected_campaign_id is not None:
    where_clauses.append(f"campaign_id = {selected_campaign_id}")


# ==============================================================
# STEP 3: PLACEMENT DRILL-DOWN (Optional)
# ==============================================================
st.markdown("---")
st.subheader("3. Placement (optional)")

selected_placement_info = None

@st.cache_data(ttl=300)
def get_placements(brand, campaign_id=None):
    """Get placements from the mapping table for selected brand/campaign."""
    brand_esc = brand.replace("'", "''")
    if campaign_id:
        where = f"CAST(m.fw_campaign_id AS BIGINT) = {campaign_id}"
    else:
        where = f"LOWER(m.locality_advertiser) LIKE LOWER('%{brand_esc}%')"
    df = run_query(f"""
        SELECT DISTINCT
            m.locality_placement_id,
            m.locality_placement_name,
            m.locality_campaign,
            m.locality_advertiser,
            m.locality_agency,
            m.advertiser_category,
            m.locality_product,
            m.locality_placement_start_date,
            m.locality_placement_end_date
        FROM {MAPPING} m
        WHERE {where}
            AND m.locality_placement_name IS NOT NULL
        ORDER BY m.locality_placement_name ASC
    """)
    return df

placements_df = get_placements(selected_brand, selected_campaign_id)

if placements_df.empty:
    st.caption("No placements found in mapping table for this selection.")
    selected_placement_label = "All Placements"
else:
    placement_labels = ["All Placements"] + [
        f"{row['locality_placement_name']} (ID: {row['locality_placement_id']})"
        for _, row in placements_df.iterrows()
    ]
    selected_placement_label = st.selectbox("Placement", placement_labels)

    if selected_placement_label != "All Placements":
        idx = placement_labels.index(selected_placement_label) - 1
        selected_placement_info = placements_df.iloc[idx].to_dict()

        # Show placement metadata
        with st.expander("Placement Details", expanded=True):
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.markdown(f"**Agency:** {selected_placement_info.get('locality_agency') or '\u2014'}")
            pcol2.markdown(f"**Product:** {selected_placement_info.get('locality_product') or '\u2014'}")
            pcol3.markdown(f"**Category:** {selected_placement_info.get('advertiser_category') or '\u2014'}")
            pcol1.markdown(f"**Start:** {selected_placement_info.get('locality_placement_start_date') or '\u2014'}")
            pcol2.markdown(f"**End:** {selected_placement_info.get('locality_placement_end_date') or '\u2014'}")
            pcol3.markdown(f"**Advertiser:** {selected_placement_info.get('locality_advertiser') or '\u2014'}")


# ==============================================================
# STEP 4: DMA SELECTION
# ==============================================================
st.markdown("---")
st.subheader("4. DMA Breakout")

@st.cache_data(ttl=300)
def get_dmas(brand, campaign_id=None):
    brand_esc = brand.replace("'", "''")
    where = f"LOWER(brand) = LOWER('{brand_esc}')"
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
    dma_list = ", ".join(f"'{d.replace(chr(39), chr(39)+chr(39))}'" for d in selected_dmas)
    where_clauses.append(f"dma IN ({dma_list})")


# ==============================================================
# STEP 5: DATE RANGE
# ==============================================================
st.markdown("---")
st.subheader("5. Date Range")
date_choice = st.radio("Analysis period", ["Full Campaign (all available data)", "Custom Date Range"], horizontal=True)

if date_choice == "Custom Date Range":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    where_clauses.append(f"report_date >= '{start_date}'")
    where_clauses.append(f"report_date <= '{end_date}'")


# ==============================================================
# GENERATE REPORT
# ==============================================================
st.markdown("---")
where_sql = " AND ".join(where_clauses)

if st.button("\U0001f4ca Generate Report", type="primary", use_container_width=True):
    with st.spinner("Querying data..."):
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

        dma_df = run_query(f"""
            SELECT dma, dma_id,
                SUM(ott_total_impressions) AS ott_impressions,
                SUM(ott_total_viewers) AS ott_viewers,
                SUM(ott_incremental_viewers) AS incremental_viewers,
                ROUND(SUM(ott_incremental_viewers) * 100.0 / NULLIF(SUM(ott_total_viewers), 0), 1) AS incrementality_pct
            FROM {VIEW}
            WHERE {where_sql}
            GROUP BY dma, dma_id
            ORDER BY incrementality_pct DESC
        """)

    # Cast Decimal types to float (Databricks SQL returns decimal.Decimal)
    for col in metrics_df.select_dtypes(include=["object"]).columns:
        try:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="ignore")
        except (TypeError, ValueError):
            pass
    for col in dma_df.columns:
        if dma_df[col].dtype == object or "decimal" in str(dma_df[col].dtype).lower():
            try:
                dma_df[col] = pd.to_numeric(dma_df[col], errors="ignore")
            except (TypeError, ValueError):
                pass

    if metrics_df.empty or metrics_df.iloc[0]["ott_total_impressions"] is None:
        st.error("No data found for the selected filters.")
        st.stop()

    m = metrics_df.iloc[0]
    pct = float(m["incrementality_pct"] or 0)

    # Store in session state for follow-up chat
    st.session_state["report_context"] = {
        "brand": selected_brand,
        "campaign": selected_campaign_label if selected_campaign_label != "All Campaigns" else "All Campaigns",
        "placement": selected_placement_info,
        "dma_count": len(selected_dmas),
        "where_sql": where_sql,
        "metrics": m.to_dict(),
        "dma_data": dma_df.to_dict(orient="records") if not dma_df.empty else [],
    }

    # ---- RENDER DASHBOARD ----
    campaign_display = st.session_state["report_context"]["campaign"]
    placement_display = ""
    if selected_placement_info:
        placement_display = f" \u2014 {selected_placement_info.get('locality_placement_name', '')}"
    st.markdown(f"### {selected_brand} \u2014 {campaign_display}{placement_display}")

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

    # Frequency Comparison — cumulative trend by day (matches iSpot dashboard)
    st.markdown("#### Media Average Frequency")
    ott_freq = float(m["ott_avg_frequency"] or 0)
    lin_freq = float(m["linear_avg_frequency"] or 0)

    # KPI row: large numbers
    kpi_l, kpi_r = st.columns(2)
    with kpi_l:
        st.markdown(
            f'<div style="padding:8px 0;">'
            f'<div style="font-size:13px; font-weight:600; color:{COLORS["navy"]};">OTT Frequency</div>'
            f'<div style="font-size:48px; font-weight:700; color:{COLORS["navy"]};">{ott_freq:.1f}</div>'
            f'</div>', unsafe_allow_html=True)
    with kpi_r:
        st.markdown(
            f'<div style="padding:8px 0;">'
            f'<div style="font-size:13px; font-weight:600; color:{COLORS["navy"]};">TV Frequency</div>'
            f'<div style="font-size:48px; font-weight:700; color:{COLORS["light_cyan"]};">{lin_freq:.1f}</div>'
            f'</div>', unsafe_allow_html=True)

    # Cumulative frequency trend by report_date from bronze table
    try:
        freq_where = f"LOWER(brand) = LOWER('{brand_escaped}')"
        if selected_campaign_id is not None:
            freq_where += f" AND campaign_id = {selected_campaign_id}"
        if selected_dmas and len(selected_dmas) < len(available_dmas):
            dma_list = ",".join(f"'{d.replace(chr(39), chr(39)+chr(39))}'" for d in selected_dmas)
            freq_where += f" AND dma IN ({dma_list})"

        freq_trend_df = run_query(f"""
            SELECT report_date,
                ROUND(SUM(ott_total_impressions) * 1.0 / NULLIF(SUM(ott_total_viewers), 0), 1) AS ott_frequency,
                ROUND(SUM(linear_total_impressions) * 1.0 / NULLIF(CAST(SUM(linear_viewers) AS BIGINT), 0), 1) AS tv_frequency
            FROM locality_dev.bronze.ispot_dma_reports_ytd
            WHERE {freq_where}
            GROUP BY report_date
            ORDER BY report_date
        """)

        if not freq_trend_df.empty and len(freq_trend_df) > 1:
            # Cast Decimal to float
            freq_trend_df["ott_frequency"] = freq_trend_df["ott_frequency"].astype(float)
            freq_trend_df["tv_frequency"] = freq_trend_df["tv_frequency"].astype(float)

            fig_freq = go.Figure()
            fig_freq.add_trace(go.Scatter(
                x=freq_trend_df["report_date"],
                y=freq_trend_df["ott_frequency"],
                name="OTT Frequency",
                mode="lines+text",
                line=dict(color=COLORS["navy"], width=2.5),
                text=freq_trend_df["ott_frequency"].apply(
                    lambda v: f"{v:.1f}" if v == freq_trend_df["ott_frequency"].iloc[-1] else ""
                ),
                textposition="top center",
                textfont=dict(size=11, color=COLORS["navy"]),
            ))
            fig_freq.add_trace(go.Scatter(
                x=freq_trend_df["report_date"],
                y=freq_trend_df["tv_frequency"],
                name="TV Frequency",
                mode="lines+text",
                line=dict(color=COLORS["light_cyan"], width=2.5),
                text=freq_trend_df["tv_frequency"].apply(
                    lambda v: f"{v:.1f}" if v == freq_trend_df["tv_frequency"].iloc[-1] else ""
                ),
                textposition="bottom center",
                textfont=dict(size=11, color=COLORS["cyan"]),
            ))
            fig_freq.update_layout(
                height=280,
                margin=dict(t=30, b=40, l=50, r=30),
                legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="white",
                yaxis=dict(title="Avg Frequency", showgrid=True, gridcolor="#eee",
                           range=[0, max(freq_trend_df["ott_frequency"].max(), freq_trend_df["tv_frequency"].max(), 1) * 1.2]),
                xaxis=dict(title=""),
                hovermode="x unified",
            )
            st.plotly_chart(fig_freq, use_container_width=True)
        else:
            st.caption("Not enough data points for frequency trend.")
    except Exception as freq_err:
        st.caption(f"Frequency trend unavailable: {freq_err}")

    # DMA Breakout
    st.markdown("#### DMA Breakout")
    if not dma_df.empty:
        # Sort by incrementality %
        display_dma = dma_df.sort_values("incrementality_pct", ascending=False).head(20)

        # Bar labels: incrementality % + impressions
        bar_labels = display_dma.apply(
            lambda r: f"{r['incrementality_pct']:.0f}%  |  {format_number(r['ott_impressions'])} imp"
            if pd.notna(r.get('incrementality_pct')) else format_number(r['ott_impressions']),
            axis=1
        )

        pct_max = float(display_dma["incrementality_pct"].max())

        fig = go.Figure(go.Bar(
            y=display_dma["dma"],
            x=display_dma["incrementality_pct"],
            orientation="h",
            marker=dict(
                color=display_dma["incrementality_pct"].tolist(),
                colorscale=[[0, COLORS["light_cyan"]], [0.5, COLORS["cyan"]], [1.0, COLORS["navy"]]],
                showscale=False,
            ),
            text=bar_labels,
            textposition="outside",
            customdata=display_dma[["ott_impressions", "ott_viewers", "incremental_viewers"]].values,
            hovertemplate=(
                "<b>%{y}</b><br><br>"
                "<b>Incrementality:</b> %{x:.1f}%<br>"
                "<span style='color:#888'>% of OTT viewers not reached by TV</span><br><br>"
                "<b>OTT Impressions:</b> %{customdata[0]:,.0f}<br>"
                "<span style='color:#888'>Total ad views delivered</span><br><br>"
                "<b>OTT Viewers:</b> %{customdata[1]:,.0f}<br>"
                "<span style='color:#888'>Unique households reached</span><br><br>"
                "<b>Incremental Viewers:</b> %{customdata[2]:,.0f}<br>"
                "<span style='color:#888'>HHs reached only via streaming</span>"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            title="Top DMAs by Incrementality %",
            xaxis=dict(title="Incrementality %", range=[0, min(pct_max * 1.3, 105)]),
            yaxis=dict(autorange="reversed"),
            margin=dict(t=40, b=30, l=180, r=140),
            height=max(300, len(display_dma) * 32),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLORS["light_gray"])
        st.plotly_chart(fig, use_container_width=True)

        # DMA Heatmap — bubble map colored by incrementality %
        if len(dma_df) > 1 and "dma_id" in dma_df.columns:
            st.markdown("#### Geography Heatmap")
            st.caption("Incrementality % by DMA region (bubble size = impressions)")
            try:
                # Nielsen DMA approximate centroids (lat, lon) by dma_id
                DMA_COORDS = {
                    500:(38.9,-77.0),501:(40.7,-74.0),502:(39.1,-76.8),503:(42.8,-73.8),504:(42.4,-71.1),
                    505:(42.9,-83.2),506:(41.5,-71.4),507:(37.3,-79.4),508:(39.3,-76.6),510:(41.5,-81.7),
                    511:(38.9,-77.0),512:(43.1,-77.6),513:(37.6,-77.5),514:(35.1,-80.8),515:(36.1,-79.8),
                    516:(36.8,-76.0),517:(35.8,-78.6),518:(35.6,-82.6),519:(32.8,-79.9),520:(33.5,-81.7),
                    521:(28.5,-81.4),522:(30.3,-81.7),523:(40.5,-74.2),524:(33.7,-84.4),525:(27.9,-82.5),
                    526:(42.1,-72.6),527:(40.8,-74.1),528:(25.8,-80.2),529:(38.6,-90.2),530:(34.7,-86.6),
                    531:(32.3,-90.2),532:(26.1,-80.1),533:(41.1,-80.8),534:(34.8,-82.4),535:(42.3,-83.0),
                    536:(37.0,-86.2),537:(40.4,-79.9),539:(27.3,-82.5),540:(28.0,-81.7),541:(37.1,-80.6),
                    542:(29.8,-95.4),543:(34.2,-77.9),544:(35.2,-81.3),545:(36.1,-80.3),546:(37.5,-77.5),
                    547:(38.3,-81.6),548:(26.7,-80.1),549:(29.3,-98.5),550:(38.0,-84.5),551:(36.2,-86.8),
                    552:(43.2,-71.5),553:(35.0,-85.3),554:(42.5,-89.0),555:(38.8,-89.6),556:(39.8,-84.2),
                    557:(36.8,-76.3),558:(35.1,-89.9),559:(35.0,-80.8),560:(33.4,-86.8),561:(30.4,-87.2),
                    563:(38.2,-85.8),564:(32.5,-84.9),565:(30.3,-87.7),566:(29.4,-98.5),567:(34.0,-81.0),
                    569:(33.3,-80.0),570:(39.1,-84.5),571:(35.6,-88.8),573:(32.5,-93.7),574:(27.8,-97.4),
                    575:(34.7,-79.9),576:(33.2,-87.5),577:(36.8,-83.3),581:(30.4,-88.9),582:(30.2,-92.0),
                    583:(32.5,-92.1),584:(35.4,-97.5),588:(36.2,-95.9),592:(34.2,-79.8),
                    # Major markets
                    602:(41.9,-87.6),603:(44.9,-93.3),604:(38.6,-90.2),605:(35.0,-85.3),
                    606:(30.5,-91.1),609:(38.2,-85.7),610:(44.5,-88.0),611:(45.5,-94.2),
                    612:(38.8,-89.6),613:(44.9,-93.3),616:(38.6,-90.2),617:(44.0,-88.5),
                    618:(29.8,-95.4),619:(42.0,-87.8),620:(42.7,-73.7),622:(30.5,-91.2),
                    623:(32.8,-96.8),624:(36.1,-95.9),625:(29.4,-98.5),626:(31.8,-106.4),
                    627:(34.7,-92.3),628:(34.0,-81.0),630:(35.1,-89.9),631:(35.1,-80.8),
                    632:(30.3,-81.7),633:(38.0,-84.5),634:(32.3,-86.3),635:(30.2,-81.7),
                    636:(29.6,-95.4),637:(38.3,-81.6),638:(34.7,-86.6),639:(30.4,-87.2),
                    640:(28.0,-81.7),641:(29.8,-90.0),642:(33.4,-86.8),643:(38.3,-85.8),
                    644:(39.8,-84.2),647:(35.6,-82.6),648:(35.8,-78.6),649:(35.0,-85.3),
                    650:(35.4,-97.5),651:(36.2,-95.9),652:(36.1,-86.8),656:(33.5,-80.8),
                    657:(35.0,-78.9),658:(36.8,-76.3),659:(37.0,-80.0),661:(37.7,-79.4),
                    662:(30.2,-92.0),669:(36.8,-83.3),670:(36.1,-79.8),671:(36.8,-76.0),
                    673:(42.3,-83.0),675:(41.1,-80.8),676:(38.6,-90.2),678:(35.0,-82.0),
                    679:(32.5,-84.9),682:(33.5,-86.8),686:(34.0,-80.9),687:(33.1,-80.0),
                    691:(34.7,-86.6),692:(34.2,-77.9),693:(36.4,-82.5),698:(32.3,-86.3),
                    # West
                    751:(38.6,-121.5),752:(37.8,-122.4),753:(34.1,-118.2),754:(47.6,-122.3),
                    755:(39.7,-104.9),756:(45.5,-122.7),757:(33.4,-112.0),758:(36.2,-115.1),
                    759:(38.8,-104.8),762:(32.7,-117.2),764:(40.8,-111.9),765:(33.4,-112.0),
                    766:(36.7,-119.8),767:(43.6,-116.2),770:(40.6,-111.9),771:(34.4,-119.7),
                    773:(44.1,-121.3),789:(47.7,-117.4),790:(46.9,-110.4),798:(48.8,-122.5),
                    800:(41.2,-111.9),801:(47.0,-122.9),802:(47.6,-117.4),803:(34.1,-118.2),
                    804:(35.4,-119.0),807:(37.8,-122.4),810:(33.9,-117.6),811:(35.3,-119.0),
                    813:(32.7,-117.2),819:(47.6,-122.3),820:(45.5,-122.7),821:(44.1,-121.3),
                    825:(38.6,-121.5),828:(36.7,-119.8),839:(36.2,-115.1),855:(39.7,-104.9),
                    862:(38.6,-121.5),866:(39.7,-104.9),868:(46.9,-114.0),881:(47.0,-117.4),
                }

                map_df = dma_df[["dma", "dma_id", "incrementality_pct", "ott_impressions", "ott_viewers", "incremental_viewers"]].copy()
                map_df["lat"] = map_df["dma_id"].map(lambda x: DMA_COORDS.get(int(x), (None, None))[0])
                map_df["lon"] = map_df["dma_id"].map(lambda x: DMA_COORDS.get(int(x), (None, None))[1])
                map_df = map_df.dropna(subset=["lat", "lon"])

                if not map_df.empty:
                    fig_map = go.Figure(go.Scattergeo(
                        lat=map_df["lat"],
                        lon=map_df["lon"],
                        marker=dict(
                            size=map_df["ott_impressions"].apply(
                                lambda x: max(8, min(40, (float(x) / map_df["ott_impressions"].max()) * 40))
                            ),
                            color=map_df["incrementality_pct"],
                            colorscale=[
                                [0, COLORS["light_cyan"]],
                                [0.4, COLORS["lime"]],
                                [0.7, COLORS["cyan"]],
                                [1.0, COLORS["navy"]],
                            ],
                            colorbar=dict(title=dict(text="Incrementality %", font=dict(size=11)), thickness=14, len=0.6, y=0.5),
                            opacity=0.8,
                            line=dict(width=0.5, color="white"),
                        ),
                        customdata=map_df[["dma", "incrementality_pct", "ott_impressions", "ott_viewers", "incremental_viewers"]].values,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br><br>"
                            "<b>Incrementality:</b> %{customdata[1]:.1f}%<br>"
                            "<span style='color:#888'>% of OTT viewers not reached by linear TV</span><br><br>"
                            "<b>OTT Impressions:</b> %{customdata[2]:,.0f}<br>"
                            "<span style='color:#888'>Total ad views delivered by Locality</span><br><br>"
                            "<b>OTT Viewers:</b> %{customdata[3]:,.0f}<br>"
                            "<span style='color:#888'>Unique households reached via streaming</span><br><br>"
                            "<b>Incremental Viewers:</b> %{customdata[4]:,.0f}<br>"
                            "<span style='color:#888'>HHs reached only by OTT (not linear TV)</span>"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    ))
                    # Size reference bubbles (legend-only, invisible on map)
                    imp_max = float(map_df["ott_impressions"].max())
                    size_refs = [
                        ("Small", 0.1, 8),
                        ("Medium", 0.4, 20),
                        ("Large", 1.0, 40),
                    ]
                    for label, frac, sz in size_refs:
                        imp_val = imp_max * frac
                        fig_map.add_trace(go.Scattergeo(
                            lat=[None], lon=[None],
                            marker=dict(size=sz, color=COLORS["mid_gray"], line=dict(width=1, color=COLORS["navy"])),
                            name=f"{label} ({format_number(imp_val)} imp)",
                            showlegend=True,
                        ))

                    fig_map.update_geos(
                        scope="usa",
                        showland=True, landcolor=COLORS["light_gray"],
                        showlakes=False,
                        showcountries=False,
                        showsubunits=True, subunitcolor="#ddd",
                    )
                    fig_map.update_layout(
                        margin=dict(t=30, b=60, l=10, r=10),
                        height=480,
                        paper_bgcolor="rgba(0,0,0,0)",
                        geo=dict(bgcolor="rgba(0,0,0,0)"),
                        legend=dict(
                            title=dict(text="Impressions", font=dict(size=11, color=COLORS["navy"])),
                            orientation="h", x=0.5, xanchor="center", y=-0.02,
                            font=dict(size=11),
                            itemsizing="constant",
                        ),
                        annotations=[
                            dict(
                                text="<b>Color</b> = Incrementality %",
                                x=0.5, y=-0.1, xref="paper", yref="paper",
                                showarrow=False, font=dict(size=11, color="#555"),
                            )
                        ],
                    )
                # Click-to-filter: capture selected DMA from map
                    event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="dma_map")

                    # Handle map click selection
                    if event and event.selection and event.selection.points:
                        clicked_idx = event.selection.points[0].get("pointIndex", None)
                        if clicked_idx is not None and clicked_idx < len(map_df):
                            clicked_row = map_df.iloc[clicked_idx]
                        clicked_dma = clicked_row["dma"]
                        clicked_pct = clicked_row["incrementality_pct"]
                        clicked_imp = clicked_row["ott_impressions"]

                        # Find full data for clicked DMA
                        dma_detail = dma_df[dma_df["dma"] == clicked_dma]
                        if not dma_detail.empty:
                            d = dma_detail.iloc[0]
                            st.markdown(
                                f'<div style="background:{COLORS["light_gray"]}; border-left:4px solid {COLORS["navy"]}; '
                                f'border-radius:4px; padding:16px 20px; margin:12px 0;">'
                                f'<span style="font-size:17px; font-weight:700; color:{COLORS["navy"]};">'
                                f'\U0001f4cd {clicked_dma}</span></div>',
                                unsafe_allow_html=True,
                            )
                            dc1, dc2, dc3, dc4 = st.columns(4)
                            dc1.metric("Incrementality", f"{clicked_pct:.1f}%")
                            dc2.metric("OTT Impressions", format_number(clicked_imp))
                            dc3.metric("OTT Viewers", format_number(d.get("ott_viewers", 0)))
                            dc4.metric("Incremental Viewers", format_number(d.get("incremental_viewers", 0)))

                            # Trend over time for clicked DMA
                            st.markdown("##### Trend Over Time")
                            try:
                                dma_esc = clicked_dma.replace("'", "''")
                                trend_where = f"LOWER(brand) = LOWER('{brand_escaped}') AND LOWER(dma) = LOWER('{dma_esc}')"
                                if selected_campaign_id is not None:
                                    trend_where += f" AND campaign_id = {selected_campaign_id}"
                                trend_df = run_query(f"""
                                    SELECT report_date,
                                        SUM(ott_total_impressions) AS ott_impressions,
                                        SUM(ott_incremental_viewers) AS incr_viewers,
                                        SUM(ott_total_viewers) AS total_viewers,
                                        ROUND(SUM(ott_incremental_viewers) * 100.0 / NULLIF(SUM(ott_total_viewers), 0), 1) AS incrementality_pct
                                    FROM locality_dev.bronze.ispot_dma_reports_ytd
                                    WHERE {trend_where}
                                    GROUP BY report_date
                                    ORDER BY report_date
                                """)

                                if not trend_df.empty and len(trend_df) > 1:
                                    # Dual-axis: incrementality % (line) + cumulative impressions (area)
                                    from plotly.subplots import make_subplots
                                    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])

                                    fig_trend.add_trace(
                                        go.Scatter(
                                            x=trend_df["report_date"],
                                            y=trend_df["incrementality_pct"],
                                            name="Incrementality %",
                                            line=dict(color=COLORS["navy"], width=2.5),
                                            mode="lines",
                                        ),
                                        secondary_y=False,
                                    )
                                    fig_trend.add_trace(
                                        go.Scatter(
                                            x=trend_df["report_date"],
                                            y=trend_df["ott_impressions"],
                                            name="OTT Impressions (cumulative)",
                                            fill="tozeroy",
                                            line=dict(color=COLORS["cyan"], width=1),
                                            fillcolor="rgba(0, 188, 212, 0.15)",
                                            mode="lines",
                                        ),
                                        secondary_y=True,
                                    )

                                    fig_trend.update_layout(
                                        height=300,
                                        margin=dict(t=20, b=40, l=50, r=50),
                                        legend=dict(orientation="h", y=-0.15),
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor=COLORS["light_gray"],
                                        hovermode="x unified",
                                    )
                                    fig_trend.update_yaxes(title_text="Incrementality %", secondary_y=False, range=[0, 100])
                                    fig_trend.update_yaxes(title_text="Impressions", secondary_y=True)
                                    st.plotly_chart(fig_trend, use_container_width=True)
                                else:
                                    st.caption("Not enough data points for trend.")
                            except Exception as trend_err:
                                st.caption(f"Trend unavailable: {trend_err}")
                    else:
                        st.caption("\U0001f446 Click a DMA region on the map to see its details.")
                else:
                    st.caption("No geographic coordinates available for selected DMAs.")

            except Exception as map_err:
                st.caption(f"Map unavailable: {map_err}")

        with st.expander(f"Full DMA Data ({len(dma_df)} DMAs)"):
            st.dataframe(dma_df, use_container_width=True)
            csv_data = dma_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="\u2B07 Download DMA Data (CSV)",
                data=csv_data,
                file_name=f"{selected_brand.replace(' ', '_')}_dma_data.csv",
                mime="text/csv",
            )

    # ---- LLM INSIGHTS NARRATIVE ----
    st.markdown("---")
    st.markdown("#### \U0001f4a1 Executive Insights")
    with st.spinner("Generating insights..."):
        insights = generate_insights(
            brand=selected_brand,
            campaign_label=campaign_display,
            metrics=m.to_dict(),
            dma_df=dma_df,
            dma_count=len(selected_dmas),
            placement_info=selected_placement_info,
        )
    st.markdown(
        f'<div style="background:{COLORS["light_gray"]}; border-left:4px solid {COLORS["cyan"]}; '
        f'border-radius:4px; padding:16px 20px; margin:8px 0; font-size:15px; line-height:1.6;">'
        f'{insights}</div>',
        unsafe_allow_html=True,
    )


    # ---- EXPORT FULL REPORT ----
    st.markdown("---")
    st.markdown("#### \U0001f4e4 Export Report")

    def build_report_csv():
        """Generate a comprehensive CSV report with all dashboard data."""
        import io
        lines = []

        # Helper to safely format values
        def _s(val):
            if val is None:
                return ""
            try:
                return str(float(val)) if hasattr(val, "as_tuple") else str(val)
            except (TypeError, ValueError):
                return str(val)

        # m is already a pandas Series (metrics_df.iloc[0]), use .get() directly
        # Section 1: Summary
        lines.append("=== SUMMARY ===")
        lines.append(f"Brand,{selected_brand}")
        lines.append(f"Campaign,{campaign_display}")
        lines.append(f"DMAs Analyzed,{len(selected_dmas)}")
        lines.append(f"Incrementality %,{_s(m.get('incrementality_pct', 0))}")
        lines.append(f"OTT Total Impressions,{_s(m.get('ott_total_impressions', 0))}")
        lines.append(f"OTT Total Viewers,{_s(m.get('ott_total_viewers', 0))}")
        lines.append(f"OTT Incremental Viewers,{_s(m.get('ott_incremental_viewers', 0))}")
        lines.append(f"All Viewers,{_s(m.get('all_viewers', 0))}")
        lines.append(f"OTT Only Impressions,{_s(m.get('ott_only_impressions', 0))}")
        lines.append(f"OTT + TV Impressions,{_s(m.get('ott_tv_impressions', 0))}")
        lines.append(f"TV Only Impressions,{_s(m.get('tv_only_impressions', 0))}")
        lines.append(f"OTT Avg Frequency,{_s(m.get('ott_avg_frequency', 0))}")
        lines.append(f"Linear Avg Frequency,{_s(m.get('linear_avg_frequency', 0))}")
        lines.append("")

        # Section 2: Placement (if selected)
        if selected_placement_info:
            pl = selected_placement_info
            lines.append("=== PLACEMENT ===")
            lines.append(f"Placement,{pl.get('locality_placement_name', '')}")
            lines.append(f"Advertiser,{pl.get('locality_advertiser', '')}")
            lines.append(f"Agency,{pl.get('locality_agency', '')}")
            lines.append(f"Product,{pl.get('locality_product', '')}")
            lines.append(f"Category,{pl.get('advertiser_category', '')}")
            lines.append(f"Start Date,{pl.get('locality_placement_start_date', '')}")
            lines.append(f"End Date,{pl.get('locality_placement_end_date', '')}")
            lines.append("")

        # Section 3: Executive Insights
        lines.append("=== EXECUTIVE INSIGHTS ===")
        lines.append(str(insights).replace(",", ";"))
        lines.append("")

        # Section 4: DMA Breakout
        if not dma_df.empty:
            lines.append("=== DMA BREAKOUT ===")
            # Convert all values to safe strings and write as CSV
            export_dma = dma_df.copy()
            for col in export_dma.columns:
                export_dma[col] = export_dma[col].apply(_s)
            csv_buf = io.StringIO()
            export_dma.to_csv(csv_buf, index=False)
            lines.append(csv_buf.getvalue())

        return "\n".join(lines).encode("utf-8")

    export_col1, export_col2 = st.columns([1, 3])
    with export_col1:
        report_bytes = build_report_csv()
        fname = f"{selected_brand.replace(' ', '_')}_{campaign_display[:20].replace(' ', '_')}_report.csv"
        st.download_button(
            label="\U0001f4ca Export Full Report",
            data=report_bytes,
            file_name=fname,
            mime="text/csv",
        )
    with export_col2:
        st.caption(
            "Exports a CSV report with: Summary Metrics, Placement Details, "
            "Executive Insights, and full DMA Breakout data."
        )


# ==============================================================
# FOLLOW-UP CHAT (persists after report generation)
# ==============================================================
if "report_context" in st.session_state:
    ctx = st.session_state["report_context"]

    st.markdown("---")
    st.markdown("#### \U0001f4ac Ask a Follow-Up Question")
    st.caption("e.g. 'Which DMA had the highest incrementality?', 'Show me only DMAs above 50%', 'How does this compare to the brand average?'")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if follow_up := st.chat_input("Ask about this campaign's performance..."):
        st.session_state["chat_history"].append({"role": "user", "content": follow_up})
        with st.chat_message("user"):
            st.markdown(follow_up)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                dma_preview = pd.DataFrame(ctx["dma_data"]).head(10).to_string(index=False) if ctx["dma_data"] else "No DMA data"
                placement_ctx = ""
                if ctx.get("placement"):
                    p = ctx["placement"]
                    placement_ctx = f"\nPlacement: {p.get('locality_placement_name', 'N/A')} | Agency: {p.get('locality_agency', 'N/A')} | Product: {p.get('locality_product', 'N/A')}"

                chat_system = f"""You are an advertising analytics expert at Locality.
The user is viewing a campaign performance report with these filters:
- Brand: {ctx['brand']}
- Campaign: {ctx['campaign']}{placement_ctx}
- DMAs: {ctx['dma_count']} markets
- SQL filter applied: WHERE {ctx['where_sql']}

Current metrics (already computed):
{pd.Series(ctx['metrics']).to_string()}

DMA breakdown (top rows):
{dma_preview}

AVAILABLE COLUMNS in {VIEW} (locality_dev.silver.ispot_dma_reports_latest):
- brand, brand_id, campaign_id, dma, dma_id, report_date
- ott_total_impressions, ott_incremental_impressions, ott_overlap_with_linear_overlap
- linear_total_impressions, linear_only_impressions
- total_impressions
- ott_total_viewers, ott_incremental_viewers, overlap_viewers, linear_viewers, all_viewers
- There is NO 'impression_type' column. The data is ALREADY split into ott_* and linear_* columns.
- To get linear impressions: use linear_total_impressions or linear_only_impressions
- To get OTT impressions: use ott_total_impressions or ott_incremental_impressions
- Incrementality = ROUND(SUM(ott_incremental_viewers) * 100.0 / NULLIF(SUM(ott_total_viewers), 0), 1)
- OTT Frequency = ROUND(SUM(ott_total_impressions) * 1.0 / NULLIF(SUM(ott_total_viewers), 0), 1)
- Linear Frequency = ROUND(SUM(linear_total_impressions) * 1.0 / NULLIF(CAST(SUM(linear_viewers) AS BIGINT), 0), 1)

RULES:
- ALWAYS provide a natural language answer that explains the insight, not just raw data. Lead with the business interpretation (e.g. "Ft. Smith has 100% incrementality because there were zero linear/TV impressions in that market — every viewer was reached exclusively through OTT streaming.").
- If you can answer from the data above, answer directly with specific numbers and explanation.
- If you need additional data, output a single SQL query wrapped in ```sql ... ``` fences FOLLOWED BY a brief explanation of what you expect the results to show. After the data is returned, the system will show it — but your text answer must still explain the meaning.
- SQL must query: {VIEW} and include WHERE {ctx['where_sql']} as a base filter.
- For mapping fields (advertiser, agency, placement), LEFT JOIN {MAPPING} ON campaign_id = CAST(fw_campaign_id AS BIGINT)
- ALWAYS use LOWER(col) LIKE '%term%' for text filters. Never use =.
- NEVER invent columns. Only use the columns listed above.
- Be concise and data-driven. Use actual numbers, not vague language.
- ALWAYS explain WHY a metric has a certain value (e.g. 100% incrementality means zero TV overlap).
- ALWAYS show incremental reach / incrementality as a PERCENTAGE (incremental_viewers / total_viewers * 100).
- When showing incrementality per DMA, format as: "DMA Name: XX.X%" (percentage first, raw counts optional in parentheses)."""

                messages = [{"role": "system", "content": chat_system}]
                messages += [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat_history"]]

                try:
                    client = get_llm_client()
                    response = client.chat.completions.create(
                        model=llm_model, messages=messages,
                        max_tokens=1000, temperature=0.2,
                    )
                    answer = response.choices[0].message.content.strip()

                    # If LLM generated SQL, execute it and show results
                    sql_match = re.search(r'```sql\s*(.+?)```', answer, re.DOTALL)
                    if sql_match:
                        extra_sql = sql_match.group(1).strip()
                        try:
                            extra_df = run_query(extra_sql)
                            if not extra_df.empty:
                                st.dataframe(extra_df, use_container_width=True)
                            answer = re.sub(r'```sql\s*.+?```', '', answer, flags=re.DOTALL).strip()
                        except Exception as sql_err:
                            answer += f"\n\n_SQL execution failed: {sql_err}_"

                    st.markdown(answer)
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})

                except Exception as e:
                    err_msg = f"Error: {e}"
                    st.error(err_msg)
                    st.session_state["chat_history"].append({"role": "assistant", "content": err_msg})
