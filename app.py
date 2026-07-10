import os
import json
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from databricks import sql as dbsql
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="iSpot Impression Agent", page_icon="\U0001f4fa", layout="wide")

# ---- Helper: get config from secrets > env > sidebar ----
def get_secret(key, default=""):
    """Read from st.secrets first, then env vars, then return default."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)

# Pre-load from secrets/.env
_host = get_secret("DATABRICKS_HOST")
_token = get_secret("DATABRICKS_TOKEN")
_warehouse = get_secret("DATABRICKS_SQL_WAREHOUSE_HTTP_PATH")
_model = get_secret("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")
_has_creds = all([_host, _token, _warehouse])

# ---- Sidebar ----
with st.sidebar:
    st.title("\U0001f4fa iSpot Agent")
    st.markdown("---")
    if _has_creds:
        st.success("Connected via secrets/env")
        db_host = _host
        db_token = _token
        db_warehouse = _warehouse
        llm_model = _model
        with st.expander("Connection Details"):
            st.text(f"Host: {db_host[:30]}...")
            st.text(f"Warehouse: ...{db_warehouse[-20:]}")
            st.text(f"Model: {llm_model}")
    else:
        st.subheader("Databricks Connection")
        db_host = st.text_input("Host", value=_host)
        db_token = st.text_input("Token", value=_token, type="password")
        db_warehouse = st.text_input("SQL Warehouse HTTP Path", value=_warehouse)
        st.markdown("---")
        st.subheader("LLM Settings")
        llm_provider = st.radio("Provider", ["Databricks FM API", "OpenAI"], index=0)
        if llm_provider == "Databricks FM API":
            llm_model = st.text_input("Model", value=_model)
        else:
            llm_model = st.text_input("Model", value="gpt-4o")
    st.markdown("---")
    st.caption("Tables: `locality_dev.bronze.ispot_dma_reports_ytd`, `locality_dev.silver.freewheel_placement_mapping`")

# Resolve LLM endpoint
if 'llm_provider' not in dir() or llm_provider == "Databricks FM API":
    llm_base_url = f"{db_host}/serving-endpoints"
    llm_api_key = db_token
else:
    llm_api_key = st.sidebar.text_input("API Key", value=get_secret("OPENAI_API_KEY"), type="password")
    llm_base_url = "https://api.openai.com/v1"

# ---- System Prompt ----
SYSTEM_PROMPT = """You are the iSpot Impression Analysis Agent for Locality.
You generate SQL against Databricks tables to answer OTT/Linear TV ad performance questions.

TABLES:
1. locality_dev.bronze.ispot_dma_reports_ytd (fact: Brand x Campaign x Date x DMA)
2. locality_dev.silver.freewheel_placement_mapping (dimension: placement metadata)
JOIN: fact.campaign_id = CAST(mapping.locality_campaign_id AS BIGINT) -- LEFT JOIN always

RULES:
- NEVER sum reach or frequency columns (ratios). Recompute: avg_freq = SUM(impressions)/SUM(viewers)
- Impressions and viewers ARE additive (safe to SUM)
- Deduplication: combined = OTT + Linear - Overlap. Use all_viewers for deduped count
- Incrementality pct = ROUND(SUM(ott_incremental_viewers) * 100.0 / NULLIF(SUM(ott_total_viewers), 0), 1) AS incrementality_pct  -- multiply by 100.0 forces DOUBLE and returns percentage directly
- Flag ott_device_count_lt_25=True or linear_device_count_lt_25=True as low-confidence
- LEFT JOIN; label unmapped as 'Unmapped'
- Fuzzy match: LOWER(col) LIKE '%term%'
- Use report_date (DATE type) for date filtering. Today is 2026-07-10
- EVERY SELECT in a UNION ALL must have its own FROM clause
- Always alias the incrementality percentage column as 'incrementality_pct' in your output
- Always include ott_incremental_viewers and ott_total_viewers alongside incrementality_pct when computing incrementality

Respond ONLY in JSON: {"thinking": "...", "sql": "...", "clarification": "...", "assumptions": "..."}"""

# ---- Locality Brand Colors ----
COLORS = {
    "navy": "#1B2A4A", "dark_blue": "#003366", "cyan": "#00BCD4",
    "light_cyan": "#80DEEA", "lime": "#C5E063", "white": "#FFFFFF",
    "light_gray": "#F8F9FA", "mid_gray": "#E0E0E0",
}


# ---- Dashboard Visualization ----
def format_number(n):
    if n is None or pd.isna(n):
        return "\u2014"
    n = float(n)
    if abs(n) >= 1e9:
        return f"{n/1e9:.1f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.1f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.0f}K"
    return f"{n:,.0f}"


def detect_viz_type(df, sql=""):
    if df is None or df.empty:
        return "empty"
    cols = set(c.lower() for c in df.columns)
    # Incrementality: match flexible column names
    has_incr = any("incremental" in c and ("pct" in c or "percent" in c or "ratio" in c) for c in cols)
    if has_incr or "incrementality_pct" in cols:
        return "incrementality"
    if "ott_pct" in cols and "linear_pct" in cols:
        return "media_buying"
    if "delivery_type" in cols and "avg_frequency" in cols:
        return "frequency_comparison"
    if "dma" in cols and len(df) > 1:
        return "geography"
    return "table"


def render_dashboard(df, sql):
    """Render dashboard visualizations based on query results."""
    viz_type = detect_viz_type(df, sql)

    if viz_type == "incrementality":
        try:
            if len(df) > 1 and "ott_total_viewers" in df.columns:
                main = df.loc[df["ott_total_viewers"].idxmax()]
            else:
                main = df.iloc[0]

            # Find incrementality pct column flexibly
            pct_col = next(
                (c for c in df.columns if "incremental" in c.lower() and ("pct" in c.lower() or "percent" in c.lower() or "ratio" in c.lower())),
                "incrementality_pct"
            )
            pct = float(main.get(pct_col, 0))
            incr = main.get("incremental_viewers", main.get("ott_incremental_viewers", 0))
            total = main.get("ott_total_viewers", 0)

            # Fix integer division: if pct=0 but viewers exist, compute client-side
            if pct == 0 and total and incr:
                pct = float(incr) / float(total) * 100
            # Handle 0-1 ratio (some SQL returns 0.25 instead of 25)
            elif 0 < pct < 1:
                pct = pct * 100

            # Confidence flag
            confidence = str(main.get("confidence", ""))
            conf_note = " (Low-Confidence)" if "low" in confidence.lower() else ""

            # Lime banner
            st.markdown(
                f'<div style="background:{COLORS["lime"]}; border-radius:8px; padding:20px 28px; margin:12px 0;">'
                f'<span style="color:{COLORS["navy"]}; font-size:20px; font-weight:700;">'
                f'{pct:.0f}% of your streaming campaign reached consumers not reached with linear TV ads.{conf_note}'
                f'</span></div>',
                unsafe_allow_html=True,
            )

            # Donut + KPI cards
            col_chart, col_kpis = st.columns([1, 3])
            with col_chart:
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
                m1, m2, m3 = st.columns(3)
                m1.metric("Incrementality", f"{pct:.1f}%")
                if total:
                    m2.metric("OTT Total Viewers", format_number(total))
                if incr:
                    m3.metric("OTT Incremental", format_number(incr))

            if len(df) > 1:
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Visualization error: {e}")
            st.dataframe(df, use_container_width=True)

    elif viz_type == "media_buying":
        row = df.iloc[0]
        ott_pct = float(row.get("ott_pct", 0))
        linear_pct = float(row.get("linear_pct", 0))
        overlap_pct = max(0, 100 - ott_pct - linear_pct)
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = go.Figure(go.Pie(
                labels=["TV Only", "OTT Only", "OTT + TV"],
                values=[linear_pct, ott_pct, overlap_pct],
                marker_colors=[COLORS["light_cyan"], COLORS["navy"], COLORS["cyan"]],
                textinfo="label+percent", textfont_size=12))
            fig.update_layout(title="OTT + TV Delivery", showlegend=False,
                margin=dict(t=40, b=20, l=10, r=10), height=280,
                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("OTT Only Impressions", format_number(row.get("ott_impressions", 0)))
            st.metric("TV Only Impressions", format_number(row.get("linear_impressions", 0)))
            st.metric("Total Impressions", format_number(row.get("total_impressions", 0)))

    elif viz_type == "frequency_comparison":
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure()
            for _, row in df.iterrows():
                dtype = row.get("delivery_type", "")
                freq = row.get("avg_frequency", 0)
                color = COLORS["navy"] if "ott" in str(dtype).lower() else COLORS["light_cyan"]
                fig.add_trace(go.Bar(x=[dtype], y=[freq], marker_color=color,
                    text=[f"{freq:.1f}"], textposition="outside",
                    textfont=dict(size=18, color=COLORS["navy"]), showlegend=False))
            fig.update_layout(title="Media Average Frequency",
                yaxis=dict(range=[0, df["avg_frequency"].max() * 1.3]),
                margin=dict(t=50, b=30, l=40, r=20), height=300,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLORS["light_gray"])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            for _, row in df.iterrows():
                st.metric(f"{row.get('delivery_type', '')} Frequency", f"{row.get('avg_frequency', 0):.1f}")

    elif viz_type == "geography":
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            dfs = df.sort_values(num_cols[0], ascending=False).head(15)
            fig = go.Figure(go.Bar(
                y=dfs["dma"], x=dfs[num_cols[0]], orientation="h",
                marker_color=COLORS["cyan"],
                text=dfs[num_cols[0]].apply(lambda x: format_number(x)),
                textposition="outside"))
            fig.update_layout(
                title=f"Top DMAs by {num_cols[0].replace('_', ' ').title()}",
                yaxis=dict(autorange="reversed"),
                margin=dict(t=50, b=30, l=150, r=60), height=max(300, len(dfs) * 32),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLORS["light_gray"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    else:
        if len(df) > 0:
            st.dataframe(df, use_container_width=True)


# ---- SQL & Agent Functions ----
def execute_sql(query):
    conn = dbsql.connect(
        server_hostname=db_host.replace("https://", "").replace("http://", ""),
        http_path=db_warehouse,
        access_token=db_token,
    )
    try:
        cur = conn.cursor()
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    finally:
        cur.close()
        conn.close()


def parse_response(raw):
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {"thinking": "parse error", "sql": None, "clarification": raw}


def empty_result_fallback(sql):
    fallback_map = {
        "advertiser_category": "SELECT DISTINCT advertiser_category FROM locality_dev.silver.freewheel_placement_mapping WHERE advertiser_category IS NOT NULL ORDER BY advertiser_category LIMIT 15",
        "locality_advertiser": "SELECT DISTINCT locality_advertiser FROM locality_dev.silver.freewheel_placement_mapping WHERE locality_advertiser IS NOT NULL ORDER BY locality_advertiser LIMIT 15",
        "locality_campaign": "SELECT DISTINCT locality_campaign FROM locality_dev.silver.freewheel_placement_mapping WHERE locality_campaign IS NOT NULL ORDER BY locality_campaign LIMIT 15",
    }
    sql_lower = sql.lower()
    for col, query in fallback_map.items():
        if col in sql_lower:
            try:
                df = execute_sql(query)
                vals = [v for v in df.iloc[:, 0].tolist() if v and str(v) != 'nan']
                name = col.replace('_', ' ').title()
                return f"No results for that {name}. Available values:\n" + "\n".join(f"  - {v}" for v in vals)
            except Exception:
                pass
    if "brand" in sql_lower and "like" in sql_lower:
        try:
            df = execute_sql("SELECT DISTINCT brand FROM locality_dev.bronze.ispot_dma_reports_ytd WHERE brand IS NOT NULL AND TRIM(brand) != '' ORDER BY brand LIMIT 15")
            vals = [v for v in df.iloc[:, 0].tolist() if v and str(v) != 'nan']
            return f"No results for that brand. Available brands:\n" + "\n".join(f"  - {v}" for v in vals)
        except Exception:
            pass
    return None


def retry_sql(client, question, bad_sql, error_msg):
    retry_prompt = f"Fix this SQL. Error: {error_msg[:300]}\nFailed SQL: {bad_sql}\nOriginal question: {question}\nREMINDER: Every SELECT in UNION ALL needs its own FROM clause."
    resp = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": retry_prompt}],
        max_tokens=2048, temperature=0.0)
    return parse_response(resp.choices[0].message.content)


def get_response(question, history):
    client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": question})

    resp = client.chat.completions.create(model=llm_model, messages=msgs, max_tokens=2048, temperature=0.0)
    parsed = parse_response(resp.choices[0].message.content)

    result = {"thinking": parsed.get("thinking"), "sql": parsed.get("sql"),
              "clarification": parsed.get("clarification"), "results": None, "answer": None, "error": None}

    if result["clarification"] and not result["sql"]:
        result["answer"] = result["clarification"]
        return result

    if result["sql"]:
        current_sql = result["sql"]
        for attempt in range(3):
            try:
                df = execute_sql(current_sql)
                result["sql"] = current_sql
                result["results"] = df
                break
            except Exception as e:
                if attempt < 2:
                    fixed = retry_sql(client, question, current_sql, str(e))
                    current_sql = fixed.get("sql") or current_sql
                else:
                    result["error"] = str(e)
                    result["answer"] = f"Query error after retries: {e}"
                    return result

        if result["results"] is not None and result["results"].empty:
            fallback = empty_result_fallback(result["sql"])
            result["answer"] = fallback or "Query returned 0 rows. Try broadening your search."
        elif result["results"] is not None:
            fmt_msg = f"Format these results concisely. Lead with numbers.\nQuestion: {question}\nSQL: {result['sql']}\nResults:\n{df.head(50).to_string(index=False)}"
            fmt = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "system", "content": "Precise data analyst. Be concise."},
                          {"role": "user", "content": fmt_msg}],
                max_tokens=1024, temperature=0.0)
            result["answer"] = fmt.choices[0].message.content
    return result


# ---- Main Chat UI ----
st.title("\U0001f4fa iSpot Impression Analysis Agent")
st.caption("Ask about OTT & Linear TV ad performance: reach, frequency, incrementality, impressions.")

if not all([db_host, db_token, db_warehouse, llm_api_key]):
    st.warning("Configure connection settings in the sidebar.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sql"):
            with st.expander("SQL Query"):
                st.code(msg["sql"], language="sql")
        if msg.get("results") is not None:
            with st.expander(f"Data ({len(msg['results'])} rows)"):
                st.dataframe(msg["results"], use_container_width=True)

if prompt := st.chat_input("Ask about impressions, reach, frequency..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            result = get_response(prompt, hist)

        if result["results"] is not None and not result["results"].empty:
            render_dashboard(result["results"], result.get("sql", ""))

        st.markdown(result["answer"] or "No response.")
        if result["sql"]:
            with st.expander("SQL Query"):
                st.code(result["sql"], language="sql")
        if result["results"] is not None:
            with st.expander(f"Raw Data ({len(result['results'])} rows)"):
                st.dataframe(result["results"], use_container_width=True)

        st.session_state.messages.append({
            "role": "assistant", "content": result["answer"],
            "sql": result.get("sql"), "results": result.get("results"),
        })
