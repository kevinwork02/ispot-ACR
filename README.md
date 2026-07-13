# iSpot Reach & Frequency Analysis

A Streamlit-based campaign performance dashboard that queries Databricks tables via SQL and uses LLM-powered insights for OTT/Linear TV advertising analysis.

## Architecture

### Three-Layer Pattern

| Layer | What it does | LLM? |
|-------|-------------|------|
| **Structured Filters** | Brand > Campaign > Placement > DMA > Date > deterministic SQL | No |
| **Dashboard** | Incrementality donut, media buying pie, frequency trend, DMA chart + map | No |
| **LLM Insights + Chat** | Executive summary + scoped follow-up Q&A with auto SQL execution | Yes |

## Data Model

| Table | Role |
|-------|------|
| `locality_dev.silver.ispot_dma_reports_latest` | **Primary** - deduplicated view, one row per brand+campaign+DMA |
| `locality_dev.bronze.ispot_dma_reports_ytd` | Raw cumulative YTD snapshots (daily, used for trend charts) |
| `locality_dev.silver.freewheel_placement_mapping` | Dimension table - placement, advertiser, agency metadata |

### Key Join
```sql
LEFT JOIN freewheel_placement_mapping m
  ON v.campaign_id = CAST(m.fw_campaign_id AS BIGINT)
```

### Key Metrics
- **Incrementality** = `ott_incremental_viewers / ott_total_viewers x 100`
- **OTT Frequency** = `ott_total_impressions / ott_total_viewers`
- **Linear Frequency** = `linear_total_impressions / linear_viewers`
- **OTT columns** = what Locality delivered (streaming)
- **Linear columns** = total TV market measured by iSpot

## Features

- Incrementality donut + KPI cards
- Media buying breakdown (OTT Only / OTT+TV / TV Only)
- Cumulative frequency trend by day (OTT vs TV lines)
- DMA bar chart sorted by incrementality (color gradient)
- DMA bubble map (size=impressions, color=incrementality, click-to-filter)
- Trend over time for clicked DMA
- LLM executive insights (3-4 sentence summary)
- Follow-up chat with auto SQL + natural language explanations
- CSV export (Summary, Placement, Insights, DMA Breakout)

### DMA Bubble Map
- Self-contained scatter_geo with ~150 embedded DMA centroids (no external GeoJSON)
- Bubble size = OTT impressions (scaled 8-40px)
- Bubble color = incrementality % (light_cyan > lime > cyan > navy)
- Click a bubble > detail card + trend-over-time chart
- Size reference legend (Small/Medium/Large) + colorbar

### Follow-up Chat
- System prompt includes full column schema (prevents hallucinated columns)
- Two-step response: SQL executes > second LLM call explains results in plain English
- Rules: NEVER invent columns, ALWAYS explain WHY

## Setup

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_TOKEN` | PAT or OAuth token |
| `DATABRICKS_SQL_WAREHOUSE_HTTP_PATH` | SQL warehouse HTTP path |
| `LLM_MODEL` | (Optional) defaults to `databricks-claude-sonnet-4-6` |

### Run Locally
```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Color Palette (Locality Brand)

| Name | Hex | Usage |
|------|-----|-------|
| Navy | `#1B2A4A` | Primary bars, text, OTT |
| Cyan | `#00BCD4` | Accent, mid-range |
| Light Cyan | `#80DEEA` | TV/Linear, low incrementality |
| Lime | `#C5E063` | Banners, mid-range on map |
| Light Gray | `#F8F9FA` | Backgrounds, cards |

## Replicating for Another Project

1. **Replace tables** - Update `VIEW` and `MAPPING` constants at top of `app.py`
2. **Update filter flow** - Modify `get_brands()`, `get_campaigns()`, `get_placements()`
3. **Update metrics query** - Change columns in main SELECT (lines ~357-374)
4. **Update DMA query** - If your data has geographic breakdowns
5. **Update chat schema** - Replace AVAILABLE COLUMNS in the chat system prompt
6. **Update insights prompt** - Adjust `generate_insights()` context

### Common Pitfalls
- **Decimal types**: Databricks SQL ROUND() returns decimal.Decimal - cast to float before Plotly
- **GeoJSON 404s**: Do not depend on external GeoJSON - embed coordinates
- **openpyxl + Decimal**: Use CSV export (openpyxl crashes with Decimal objects)
- **LLM column hallucination**: Include full column list + NEVER invent columns rule
- **Series.iloc[0]**: If m = df.iloc[0] (already a Series), do not call m.iloc[0] again

## License

Internal use - Locality.