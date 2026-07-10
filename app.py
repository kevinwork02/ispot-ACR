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
llm_model = get_secret("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")

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
            COALESCE(MAX(m.locality_campaign), CONCAT('Campaign ', v.campaign_id)) AS campaign_name,
            SUM(v.ott_total_impressions) AS ott_imp
        FROM {VIEW} v
        LEFT JOIN {MAPPING} m
            ON v.campaign_id = CAST(m.locality_campaign_id AS BIGINT)
        WHERE LOWER(v.brand) = LOWER('{brand_esc}')
        GROUP BY v.campaign_id
        ORDER BY ott_imp DESC
    """)
    return df

campaigns_df = get_campaigns(selected_brand)
campaign_labels = ["All Campaigns"] + [
    f"{row['campaign_name']} (ID: {row['campaign_id']})"
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
        where = f"CAST(m.locality_campaign_id AS BIGINT) = {campaign_id}"
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
            dma_count=len(selected_dmas),
            placement_info=selected_placement_info,
        )
    st.markdown(
        f'<div style="background:{COLORS["light_gray"]}; border-left:4px solid {COLORS["cyan"]}; '
        f'border-radius:4px; padding:16px 20px; margin:8px 0; font-size:15px; line-height:1.6;">'
        f'{insights}</div>',
        unsafe_allow_html=True,
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

RULES:
- If you can answer from the data above, answer directly with specific numbers.
- If you need additional data, output a single SQL query wrapped in ```sql ... ``` fences.
- SQL must query: {VIEW} and include WHERE {ctx['where_sql']} as a base filter.
- For mapping fields (advertiser, agency, placement), LEFT JOIN {MAPPING} ON campaign_id = CAST(locality_campaign_id AS BIGINT)
- ALWAYS use LOWER(col) LIKE '%term%' for text filters. Never use =.
- Be concise and data-driven. Use actual numbers, not vague language."""

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
