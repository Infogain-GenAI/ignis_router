"""
Streamlit dashboard for validating Ignis Router dashboard metrics.

Reads data from the FastAPI /dashboard endpoint (not directly from the DB).

Usage:
    1. Start the API:   python -m ignis_router.api.run_api
    2. Run dashboard:   python -m streamlit run examples/streamlit_dashboard.py
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


WINDOW_OPTIONS = {
    "Last 24h": 1,
    "Last 7d": 7,
    "Last 30d": 30,
    "Last 90d": 90,
}
PAGE_SIZE = 20

# Base URL of the running Ignis Router API
API_BASE_URL = os.getenv("IGNIS_API_URL", "http://127.0.0.1:8080")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _format_money(value: float) -> str:
    return f"${value:,.6f}" if value < 1 else f"${value:,.2f}"


def _format_pct(value: float) -> str:
    return f"{value:.1f}%"


def _format_latency(value: float) -> str:
    return f"{value:.0f} ms"


def _build_metric_cards(kpis: dict) -> list[tuple[str, str, str]]:
    return [
        ("QUERIES", str(kpis.get("query_count", 0)), "Filtered query count"),
        (
            "ROUTING ACCURACY",
            _format_pct(kpis.get("routing_accuracy_pct", 0)),
            "Confidence proxy from current slice",
        ),
        (
            "COST SAVINGS",
            _format_pct(kpis.get("cost_savings_pct", 0)),
            f"Total cost {_format_money(kpis.get('total_cost_usd', 0))}",
        ),
        (
            "AVG LATENCY",
            _format_latency(kpis.get("avg_routing_latency_ms", 0)),
            f"P95 {_format_latency(kpis.get('p95_routing_latency_ms', 0))}",
        ),
        (
            "ML WIN RATE",
            _format_pct(kpis.get("ml_win_rate_pct", 0)),
            f"Avg confidence {kpis.get('avg_confidence', 0):.2f} (model-based)",
        ),
    ]


@st.cache_data(ttl=30, show_spinner=False)
def _load_dashboard(days: int, strategy: str | None, intent: str | None, page: int) -> dict:
    """Call the FastAPI /dashboard endpoint with the selected filters."""
    params: dict[str, str | int] = {"days": days, "page": page, "page_size": PAGE_SIZE}
    if strategy:
        params["strategy"] = strategy
    if intent:
        params["intent"] = intent

    response = httpx.get(f"{API_BASE_URL}/dashboard", params=params, timeout=15.0)
    response.raise_for_status()
    return response.json()


def _to_dataframe(items: list[dict], rename: dict[str, str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(items)
    if rename and not frame.empty:
        frame = frame.rename(columns=rename)
    return frame


def _render_top_bar(payload: dict, selected_window: str, strategies: list[str], intents: list[str]) -> tuple[int, str | None, str | None]:
    with st.container(border=True):
        header_left, header_right = st.columns([8, 1])
        header_left.markdown("## IGNIS ROUTER DASHBOARD")
        header_right.markdown("### v0.1.0")

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 2, 2, 1])
        with filter_col1:
            window_label = st.selectbox("Window", options=list(WINDOW_OPTIONS.keys()), index=list(WINDOW_OPTIONS.keys()).index(selected_window))
        with filter_col2:
            strategy_label = st.selectbox("Strategy", options=["All", *strategies])
        with filter_col3:
            intent_label = st.selectbox("Intent", options=["All", *intents])
        with filter_col4:
            st.write("")
            if st.button("Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        start_text = payload.get("start_date", "")
        end_text = payload.get("end_date", "")
        st.caption(f"Range: {start_text}  ->  {end_text}")

    strategy_value = None if strategy_label == "All" else strategy_label
    intent_value = None if intent_label == "All" else intent_label
    return WINDOW_OPTIONS[window_label], strategy_value, intent_value


def _render_metrics(kpis: dict) -> None:
    metric_columns = st.columns(5)
    for column, (label, value, help_text) in zip(metric_columns, _build_metric_cards(kpis)):
        with column:
            st.metric(label=label, value=value, help=help_text)


def _render_chart_card(title: str, data: pd.DataFrame, x: str, y: str, *, horizontal: bool = False) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if data.empty:
            st.info("No data available for this filter selection.")
            return
        chart_data = data.set_index(x)[[y]]
        if horizontal:
            st.dataframe(data, use_container_width=True, hide_index=True)
        else:
            st.bar_chart(chart_data, use_container_width=True)


def _render_line_card(title: str, data: pd.DataFrame, x: str, y: str) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if data.empty:
            st.info("No data available for this filter selection.")
            return
        chart_data = data.set_index(x)[[y]]
        st.line_chart(chart_data, use_container_width=True)


def _render_intent_metrics_section(payload: dict) -> None:
    """Render all intent-related metrics grouped into one section."""
    st.markdown("---")
    st.markdown("### 🎯 INTENT METRICS")
    st.caption("All intent-related data — distribution, confidence scores, and routing outcomes per intent.")

    dist_col, perf_col = st.columns(2)
    with dist_col:
        _render_chart_card(
            "INTENT DISTRIBUTION",
            _to_dataframe(payload["charts"].get("intent_distribution", [])),
            "intent",
            "query_count",
            horizontal=True,
        )

    intent_df = _to_dataframe(
        payload["charts"].get("score_by_intent", []),
        {
            "intent": "Intent",
            "query_count": "Queries",
            "avg_confidence": "Avg Conf",
            "top_model": "Top Model",
            "ml_won_count": "ML Won",
            "rule_based_won_count": "Rule Won",
        },
    )
    if not intent_df.empty:
        if "ML Won" in intent_df.columns and "Rule Won" in intent_df.columns:
            intent_df["ML vs Rule"] = intent_df.apply(
                lambda row: f"ML: {int(row['ML Won'])}  Rule: {int(row['Rule Won'])}",
                axis=1,
            )
            intent_df = intent_df[["Intent", "Queries", "Avg Conf", "Top Model", "ML vs Rule"]]
        else:
            intent_df = intent_df[[col for col in ["Intent", "Queries", "Avg Conf", "Top Model"] if col in intent_df.columns]]
    with perf_col:
        with st.container(border=True):
            st.markdown("#### PERFORMANCE BY INTENT")
            st.dataframe(intent_df, use_container_width=True, hide_index=True)


def _render_ml_vs_rule(payload: dict) -> None:
    chart = payload["charts"].get("ml_vs_rulebased", {})
    ml_won = int(chart.get("ml_won", 0))
    rule_won = int(chart.get("rule_based_won", 0))
    total = max(ml_won + rule_won, 1)
    display_df = pd.DataFrame(
        [
            {"Outcome": "ML Won", "Count": ml_won, "Pct": round(ml_won / total * 100, 2)},
            {"Outcome": "Rule-Based Won", "Count": rule_won, "Pct": round(rule_won / total * 100, 2)},
        ]
    )
    with st.container(border=True):
        st.markdown("#### ML vs RULE-BASED")
        left, right = st.columns([2, 1])
        with left:
            st.bar_chart(display_df.set_index("Outcome")[["Count"]], use_container_width=True)
        with right:
            st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_model_metrics_section(payload: dict) -> None:
    """Render all model-related metrics grouped into one section."""
    st.markdown("---")
    st.markdown("### 🤖 MODEL METRICS")
    st.caption("All model-related performance data — distribution, confidence scores (model-based), and routing outcomes.")

    dist_col, conf_col = st.columns(2)
    with dist_col:
        _render_chart_card(
            "MODEL DISTRIBUTION",
            _to_dataframe(payload["charts"].get("model_distribution", [])),
            "model",
            "query_count",
            horizontal=True,
        )
    with conf_col:
        _render_chart_card(
            "CONFIDENCE DISTRIBUTION (model-based)",
            _to_dataframe(payload["charts"].get("confidence_distribution", [])),
            "range",
            "pct",
            horizontal=True,
        )

    model_df = _to_dataframe(
        payload["charts"].get("score_by_model", []),
        {
            "model": "Model",
            "query_count": "Queries",
            "avg_confidence": "Avg Conf (model-based)",
            "avg_cost_usd": "Avg Cost",
            "avg_latency_ms": "Avg Latency",
        },
    )
    with st.container(border=True):
        st.markdown("#### PERFORMANCE BY MODEL")
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    _render_ml_vs_rule(payload)


def _render_routing_log(payload: dict) -> None:
    pagination = payload.get("routing_log_pagination", {})
    current_page = int(pagination.get("page", 1))
    total_pages = int(pagination.get("total_pages", 1))

    log_df = _to_dataframe(
        payload.get("routing_log", []),
        {
            "created_at": "Time",
            "query": "Query",
            "intent": "Intent",
            "model": "Model",
            "provider": "Provider",
            "confidence": "Conf",
            "cost_usd": "Cost",
            "latency_ms": "Latency",
            "strategy": "Strategy",
            "complexity": "Complexity",
            "ml_won": "ML Won",
        },
    )
    if not log_df.empty:
        preferred_columns = [
            "Time",
            "Query",
            "Intent",
            "Model",
            "Conf",
            "Cost",
            "Latency",
            "Provider",
            "Strategy",
            "Complexity",
            "ML Won",
        ]
        log_df = log_df[[column for column in preferred_columns if column in log_df.columns]]

    with st.container(border=True):
        title_col, page_col = st.columns([6, 1])
        title_col.markdown("#### RECENT ROUTING DECISIONS")
        page_col.markdown(f"**Page {current_page}/{total_pages}**")
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        prev_col, spacer_col, next_col = st.columns([1, 6, 1])
        with prev_col:
            if st.button("◀ Prev", disabled=not pagination.get("has_prev", False), use_container_width=True):
                st.session_state["dashboard_page"] = max(current_page - 1, 1)
                st.rerun()
        with next_col:
            if st.button("Next ▶", disabled=not pagination.get("has_next", False), use_container_width=True):
                st.session_state["dashboard_page"] = current_page + 1
                st.rerun()


def main() -> None:
    load_dotenv(_project_root() / ".env")

    st.set_page_config(
        page_title="Ignis Router Dashboard",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        h2, h3, h4, label, [data-testid="stMetricLabel"] {
            color: #f8fafc !important;
        }
        [data-testid="stMetricValue"] {
            color: #f59e0b;
        }
        .stCaption, p, div {
            color: #cbd5e1;
        }
        /* Make ALL selectbox/dropdown text black */
        .stSelectbox div,
        .stSelectbox span,
        .stSelectbox input,
        .stSelectbox [data-baseweb="select"] *,
        [data-baseweb="popover"] *,
        [data-baseweb="menu"] *,
        [data-baseweb="select"] *,
        ul[role="listbox"] *,
        [role="listbox"] *,
        [role="option"] *,
        [role="option"],
        [data-baseweb="menu"],
        [data-baseweb="popover"] {
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "dashboard_page" not in st.session_state:
        st.session_state["dashboard_page"] = 1
    if "window_label" not in st.session_state:
        st.session_state["window_label"] = "Last 24h"

    selected_window = st.session_state["window_label"]
    selected_days = WINDOW_OPTIONS[selected_window]

    try:
        base_payload = _load_dashboard(days=selected_days, strategy=None, intent=None, page=1)
    except httpx.ConnectError:
        st.error(f"Cannot connect to Ignis Router API at {API_BASE_URL}. Is the API running?")
        st.info("Start it with:  python -m ignis_router.api.run_api")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load dashboard data from API: {exc}")
        st.stop()

    strategies = base_payload.get("kpis", {}).get("strategies_used", [])
    intents = base_payload.get("kpis", {}).get("intents_detected", [])
    selected_days, selected_strategy, selected_intent = _render_top_bar(
        base_payload,
        selected_window,
        strategies,
        intents,
    )

    current_signature = (selected_days, selected_strategy, selected_intent)
    if st.session_state.get("dashboard_signature") != current_signature:
        st.session_state["dashboard_signature"] = current_signature
        st.session_state["dashboard_page"] = 1

    page = int(st.session_state.get("dashboard_page", 1))
    payload = _load_dashboard(
        days=selected_days,
        strategy=selected_strategy,
        intent=selected_intent,
        page=page,
    )

    st.session_state["window_label"] = next(
        label for label, days in WINDOW_OPTIONS.items() if days == selected_days
    )

    _render_metrics(payload.get("kpis", {}))

    row1_left, row1_right = st.columns([1, 1])
    with row1_left:
        _render_chart_card(
            "QUERIES OVER TIME",
            _to_dataframe(payload["charts"].get("queries_over_time", [])),
            "date",
            "query_count",
        )

    _render_line_card(
        "COST TREND",
        _to_dataframe(payload["charts"].get("cost_trend", [])),
        "date",
        "total_cost_usd",
    )

    row3_left, row3_right = st.columns(2)
    with row3_left:
        _render_line_card(
            "LATENCY TREND",
            _to_dataframe(payload["charts"].get("latency_trend", [])),
            "date",
            "avg_latency_ms",
        )
    with row3_right:
        _render_line_card(
            "ACCURACY OVER TIME",
            _to_dataframe(payload["charts"].get("accuracy_over_time", [])),
            "date",
            "intent_accuracy_pct",
        )

    _render_model_metrics_section(payload)
    _render_intent_metrics_section(payload)
    _render_routing_log(payload)

    st.divider()
    _render_feature_flags(payload)

    st.divider()
    _render_query_tester()


def _render_feature_flags(payload: dict) -> None:
    """Render feature flag toggles that call PUT /features/{key} on change."""
    with st.container(border=True):
        st.markdown("#### ⚙️ FEATURE FLAGS")
        st.caption("Toggle routing features on/off in real-time (no restart needed)")

        feature_flags = payload.get("feature_flags", {})
        available = feature_flags.get("available_keys", [])

        if not available:
            st.warning("No feature flags available from API.")
            return

        # Map internal keys to clean API dropdown keys
        key_to_api = {
            "enable_ml_model_hint_routing": "ml_based_routing",
            "enable_rule_based_intent_detection": "rule_based_routing",
            "enable_ml_intent_detection": "hybrid_routing",
        }

        cols = st.columns(len(available))
        for col, flag in zip(cols, available):
            with col:
                current = flag["enabled"]
                new_value = st.toggle(
                    flag["name"],
                    value=current,
                    key=f"flag_{flag['key']}",
                    help=f"Config: {flag['key']}",
                )
                if new_value != current:
                    api_key = key_to_api.get(flag["key"], flag["key"])
                    try:
                        resp = httpx.put(
                            f"{API_BASE_URL}/features/{api_key}",
                            params={"enabled": new_value},
                            timeout=5.0,
                        )
                        if resp.status_code == 200:
                            st.success(f"{'Enabled' if new_value else 'Disabled'} ✓")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"Failed: {resp.text}")
                    except Exception as exc:
                        st.error(f"API error: {exc}")


def _render_query_tester() -> None:
    """Render a query testing tab to test routing and chat APIs."""
    with st.container(border=True):
        st.markdown("#### 🧪 QUERY TESTER")
        st.caption("Test routing and chat APIs directly from the dashboard")

        test_tab1, test_tab2 = st.tabs(["Route (pick model)", "Chat (route + LLM)"])

        with test_tab1:
            query_route = st.text_input(
                "Enter query to route:",
                placeholder="e.g. Write Python code for sorting",
                key="route_query_input",
            )
            if st.button("Route", key="btn_route", use_container_width=True):
                if query_route.strip():
                    with st.spinner("Routing..."):
                        try:
                            resp = httpx.post(
                                f"{API_BASE_URL}/route",
                                json={"query": query_route.strip()},
                                timeout=30.0,
                            )
                            if resp.status_code == 200:
                                result = resp.json()
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Selected Model", result.get("selected_model", "—"))
                                col2.metric("Strategy", result.get("strategy", "—"))
                                col3.metric("Confidence (model-based)", f"{result.get('confidence', 0):.2f}")
                                st.json(result)
                            else:
                                st.error(f"Error {resp.status_code}: {resp.text}")
                        except Exception as exc:
                            st.error(f"Request failed: {exc}")
                else:
                    st.warning("Please enter a query.")

        with test_tab2:
            query_chat = st.text_area(
                "Enter query for chat:",
                placeholder="e.g. Explain how binary search works",
                key="chat_query_input",
                height=100,
            )
            chat_col1, chat_col2 = st.columns(2)
            with chat_col1:
                system_prompt = st.text_input(
                    "System prompt:",
                    value="You are a helpful assistant.",
                    key="chat_system_prompt",
                )
            with chat_col2:
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="chat_temp")

            if st.button("Send to LLM", key="btn_chat", use_container_width=True):
                if query_chat.strip():
                    with st.spinner("Routing + calling LLM..."):
                        try:
                            resp = httpx.post(
                                f"{API_BASE_URL}/chat",
                                json={
                                    "query": query_chat.strip(),
                                    "system_prompt": system_prompt,
                                    "temperature": temperature,
                                    "max_tokens": 1024,
                                },
                                timeout=60.0,
                            )
                            if resp.status_code == 200:
                                result = resp.json()
                                # Show routing decision
                                rd = result.get("routing_decision", {})
                                rc1, rc2, rc3, rc4 = st.columns(4)
                                rc1.metric("Model", result.get("model", "—"))
                                rc2.metric("Provider", result.get("provider", "—"))
                                rc3.metric("Tokens", rd.get("tokens", 0))
                                rc4.metric("Time", f"{rd.get('time', 0):.2f}s")

                                if rd.get("ml_router_predicted"):
                                    st.info(f"🤖 ML predicted: **{rd['ml_router_predicted']}**")
                                if rd.get("note"):
                                    st.warning(f"⚠️ {rd['note']}")

                                # Show response
                                st.markdown("---")
                                st.markdown("**Response:**")
                                st.markdown(result.get("content", "No response"))

                                # Expandable full JSON
                                with st.expander("Full API response"):
                                    st.json(result)
                            else:
                                st.error(f"Error {resp.status_code}: {resp.text}")
                        except httpx.ReadTimeout:
                            st.error("Request timed out (LLM took too long)")
                        except Exception as exc:
                            st.error(f"Request failed: {exc}")
                else:
                    st.warning("Please enter a query.")


if __name__ == "__main__":
    main()