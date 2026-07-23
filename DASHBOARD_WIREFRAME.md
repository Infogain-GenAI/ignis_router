# Dashboard UI Wireframe — Routing Intelligence Dashboard

## Overview

The dashboard is structured as:
**Time Window Selector → KPI Cards → Charts Grid → Routing Log Table**

---

## Full Wireframe

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  🔥  IGNIS ROUTER DASHBOARD                                          v0.1.0    ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Window : [ Last 24h ▼ ]    Strategy : [ All ▼ ]    Intent : [ All ▼ ]         ║
║  Range  : 2026-07-22 10:00 UTC  ──────────────────►  2026-07-23 10:00 UTC      ║
║                                                               [ 🔄 Refresh ]    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│  QUERIES   │  │  ROUTING   │  │   COST     │  │  AVG       │  │  ML WIN    │
│            │  │  ACCURACY  │  │  SAVINGS   │  │  LATENCY   │  │  RATE      │
│    147     │  │   88.2%    │  │   32.4%    │  │  42 ms     │  │   66.7%    │
│  ▲ +23%   │  │  ▲ +5.1%   │  │  ▲ +8.2%  │  │  ▼ -12ms  │  │  ▲ +4.2%  │
└────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘

┌─────────────────────────────────┐  ┌─────────────────────────────────────────┐
│  QUERIES OVER TIME              │  │  CONFIDENCE DISTRIBUTION                │
│                                 │  │                                         │
│  50│        ╭──╮               │  │  High (≥0.80)   ██████████████░░  62%   │
│  40│   ╭──╮ │  │  ╭──╮        │  │  Med (0.60-79)  ████████░░░░░░░░  27%   │
│  30│╭─╮│  │ │  │  │  │╭──╮   │  │  Low (<0.60)    ████░░░░░░░░░░░░  11%   │
│  20││ ││  │ │  │╭╮│  ││  │   │  │                                         │
│  10││ ││  │ │  ││││  ││  │   │  │                                         │
│   0│┴─┴┴──┴─┴──┴┴┴┴──┴┴──┴   │  │                                         │
│    Mon Tue Wed Thu Fri Sat Sun │  │                                         │
└─────────────────────────────────┘  └─────────────────────────────────────────┘

┌─────────────────────────────────┐  ┌─────────────────────────────────────────┐
│  MODEL DISTRIBUTION             │  │  INTENT DISTRIBUTION                    │
│                                 │  │                                         │
│  gpt-4o-mini     ████████░  42% │  │  code_generation  █████████░  38%      │
│  claude-3-5      ██████░░░  31% │  │  general_chat     ██████░░░░  22%      │
│  gpt-4.1         ████░░░░░  19% │  │  reasoning        █████░░░░░  18%      │
│  gemini-1.5      ██░░░░░░░   8% │  │  summarization    ███░░░░░░░  12%      │
│                                 │  │  creative_writing  ██░░░░░░░░   6%      │
│                                 │  │  other             █░░░░░░░░░   4%      │
└─────────────────────────────────┘  └─────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  COST TREND                                                                      │
│                                                                                  │
│  $0.05│                    ╭╮                                                   │
│  $0.04│              ╭──╮  ││                                                   │
│  $0.03│         ╭──╮ │  │  ││  ╭──╮                                            │
│  $0.02│    ╭──╮ │  │ │  │  ││  │  │  ╭──╮                                     │
│  $0.01│╭─╮ │  │ │  │ │  │╭╯╰╮ │  │  │  │                                     │
│  $0.00│┴─┴─┴──┴─┴──┴─┴──┴┴──┴─┴──┴──┴──┴                                     │
│       Mon  Tue  Wed  Thu  Fri  Sat  Sun                                         │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  PERFORMANCE BY MODEL                                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Model             Queries  Avg Conf  Avg Cost    Avg Latency                   │
│  ──────────────────────────────────────────────────────────────                  │
│  gpt-4o-mini         62      0.82     $0.000045     28 ms                       │
│  claude-3-5-sonnet   45      0.91     $0.001200     35 ms                       │
│  gpt-4.1             28      0.87     $0.003200     42 ms                       │
│  gemini-1.5-pro      12      0.78     $0.000500     31 ms                       │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  PERFORMANCE BY INTENT                                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Intent              Queries  Avg Conf  Top Model           ML vs Rule           │
│  ──────────────────────────────────────────────────────────────────────          │
│  code_generation       56      0.89     claude-3-5-sonnet   ML: 48  Rule: 8     │
│  general_chat          32      0.74     gpt-4o-mini         ML: 18  Rule: 14    │
│  reasoning             26      0.85     gpt-4.1             ML: 22  Rule: 4     │
│  summarization         18      0.91     gpt-4.1             ML: 16  Rule: 2     │
│  creative_writing       9      0.88     claude-3-5-sonnet   ML:  8  Rule: 1     │
│  data_analysis          6      0.82     gpt-4.1             ML:  5  Rule: 1     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  ML vs RULE-BASED                                                                │
│                                                                                  │
│  ML Won          ████████████████████████████████████░░░░░░░░░░  72%  (106)     │
│  Rule-Based Won  ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  28%   (41)     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  RECENT ROUTING DECISIONS                                              Page 1/5 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Time       Query                          Intent          Model        Conf  $  │
│  ─────────────────────────────────────────────────────────────────────────────   │
│  10:28:00   what is json?                  data_analysis   gpt-4.1      0.60  …  │
│  10:12:34   how we can make maggie         general_chat    gpt-4.1      0.60  …  │
│  10:07:44   summarise the code.            summarization   gpt-4.1      0.72  …  │
│  09:36:09   what is aqi of delhi today     general_chat    gpt-4.1      0.60  …  │
│  09:35:44   write a code for 2sum problem  code_generation gpt-4.1      0.40  …  │
│  09:32:06   who is pm of india?            general_chat    gpt-4.1      0.60  …  │
│                                                                                  │
│  [ ◀ Prev ]                                                    [ Next ▶ ]       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Reference

### 1. Top Bar

```
╔══════════════════════════════════════════════════════╗
║  Window: [Last 24h ▼]  Strategy: [All ▼]            ║
║  Intent: [All ▼]                    [Refresh 🔄]     ║
╚══════════════════════════════════════════════════════╝
```

| Element | API Field | Behaviour |
|---|---|---|
| Window picker | `window_hours` | 1h / 6h / 24h / 7d / 30d → sets `days` param |
| Strategy filter | `kpis.strategies_used` | Filter by routing strategy |
| Intent filter | `kpis.intents_detected` | Filter by detected intent |
| Refresh | — | Re-fires `GET /dashboard?days=N` |

---

### 2. KPI Cards Row

```
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│  QUERIES   │  │  ROUTING   │  │   COST     │  │  AVG       │  │  ML WIN    │
│   147      │  │  ACCURACY  │  │  SAVINGS   │  │  LATENCY   │  │  RATE      │
│            │  │   88.2%    │  │   32.4%    │  │   42 ms    │  │   66.7%    │
└────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘
```

| Element | API Field |
|---|---|
| Total queries | `kpis.query_count` |
| Routing accuracy | `kpis.routing_accuracy_pct` |
| Cost savings | `kpis.cost_savings_pct` |
| Total cost | `kpis.total_cost_usd` |
| Avg cost per query | `kpis.avg_cost_per_query_usd` |
| Unnecessary premium usage | `kpis.unnecessary_premium_pct` |
| Avg latency | `kpis.avg_routing_latency_ms` |
| P95 latency | `kpis.p95_routing_latency_ms` |
| Avg confidence | `kpis.avg_confidence` |
| ML win rate | `kpis.ml_win_rate_pct` |
| Intent accuracy | `kpis.intent_accuracy_pct` |
| Top-2 accuracy | `kpis.top2_accuracy_pct` |

---

### 3. Charts Grid

| Chart | Type | API Field |
|---|---|---|
| Queries Over Time | Bar chart | `charts.queries_over_time[].date`, `.query_count` |
| Confidence Distribution | Horizontal bars | `charts.confidence_distribution[].range`, `.count`, `.pct` |
| Model Distribution | Horizontal bars / Donut | `charts.model_distribution[].model`, `.query_count`, `.pct` |
| Intent Distribution | Horizontal bars / Donut | `charts.intent_distribution[].intent`, `.query_count`, `.pct` |
| Cost Trend | Line chart | `charts.cost_trend[].date`, `.total_cost_usd` |
| Latency Trend | Line chart | `charts.latency_trend[].date`, `.avg_latency_ms` |
| Accuracy Over Time | Line chart | `charts.accuracy_over_time[].date`, `.intent_accuracy_pct` |
| ML vs Rule-Based | Stacked bar | `charts.ml_vs_rulebased.ml_won`, `.rule_based_won` |

---

### 4. Performance By Model Table

```
Model             Queries  Avg Conf  Avg Cost    Avg Latency
gpt-4o-mini         62      0.82     $0.000045     28 ms
claude-3-5-sonnet   45      0.91     $0.001200     35 ms
```

| Element | API Field |
|---|---|
| Model name | `charts.score_by_model[].model` |
| Query count | `charts.score_by_model[].query_count` |
| Avg confidence | `charts.score_by_model[].avg_confidence` |
| Avg cost | `charts.score_by_model[].avg_cost_usd` |
| Avg latency | `charts.score_by_model[].avg_latency_ms` |

---

### 5. Performance By Intent Table

```
Intent              Queries  Avg Conf  Top Model
code_generation       56      0.89     claude-3-5-sonnet
general_chat          32      0.74     gpt-4o-mini
```

| Element | API Field |
|---|---|
| Intent name | `charts.score_by_intent[].intent` |
| Query count | `charts.score_by_intent[].query_count` |
| Avg confidence | `charts.score_by_intent[].avg_confidence` |
| Top model | `charts.score_by_intent[].top_model` |

---

### 6. Routing Log Table

```
Time       Query                     Intent         Model      Conf   Cost
10:28:00   what is json?             data_analysis  gpt-4.1    0.60   $0.003
```

| Element | API Field |
|---|---|
| Timestamp | `routing_log[].created_at` |
| Query text | `routing_log[].query` |
| Intent | `routing_log[].intent` |
| Model | `routing_log[].model` |
| Provider | `routing_log[].provider` |
| Confidence | `routing_log[].confidence` |
| Latency | `routing_log[].latency_ms` |
| Cost | `routing_log[].cost_usd` |
| ML won | `routing_log[].ml_won` |
| Strategy | `routing_log[].strategy` |
| Complexity | `routing_log[].complexity` |

---

### 7. States to Handle

| State | UI Behaviour |
|---|---|
| No queries in window | Show "No routing decisions in this time window" |
| DB not connected | Show banner: "Database not connected" |
| Latency = 0 | Show "—" (old data before tracking) |
| Cost = 0 | Show "—" (old data before tracking) |
| Confidence < 0.60 | Red text / warning icon |
| Confidence ≥ 0.80 | Green text / success icon |
| ML won = true | Blue badge "ML" |
| ML won = false | Grey badge "Rule" |
| `unnecessary_premium_pct > 5%` | Warning card highlight |

---

## API Contract Summary

### `GET /dashboard?days=7`

Single endpoint — returns everything the frontend needs.

```json
{
  "generated_at":  "<ISO>",
  "window_hours":  "<int>",
  "start_date":    "<ISO>",
  "end_date":      "<ISO>",

  "kpis": {
    "query_count":              "<int>",
    "routing_accuracy_pct":     "<float>",
    "intent_accuracy_pct":      "<float>",
    "top2_accuracy_pct":        "<float>",
    "cost_savings_pct":         "<float>",
    "total_cost_usd":           "<float>",
    "avg_cost_per_query_usd":   "<float>",
    "unnecessary_premium_pct":  "<float>",
    "avg_routing_latency_ms":   "<float>",
    "p95_routing_latency_ms":   "<float>",
    "avg_confidence":           "<float>",
    "ml_win_rate_pct":          "<float>",
    "models_used":              ["<string>"],
    "strategies_used":          ["<string>"],
    "intents_detected":         ["<string>"]
  },

  "charts": {
    "accuracy_over_time": [
      { "date": "<YYYY-MM-DD>", "query_count": "<int>",
        "avg_confidence": "<float>", "intent_accuracy_pct": "<float>" }
    ],
    "model_distribution": [
      { "model": "<string>", "query_count": "<int>", "pct": "<float>" }
    ],
    "intent_distribution": [
      { "intent": "<string>", "query_count": "<int>", "pct": "<float>" }
    ],
    "cost_trend": [
      { "date": "<YYYY-MM-DD>", "total_cost_usd": "<float>" }
    ],
    "latency_trend": [
      { "date": "<YYYY-MM-DD>", "avg_latency_ms": "<float>",
        "query_count": "<int>" }
    ],
    "confidence_distribution": [
      { "range": "<string>", "count": "<int>", "pct": "<float>" }
    ],
    "score_by_model": [
      { "model": "<string>", "avg_confidence": "<float>",
        "avg_cost_usd": "<float>", "avg_latency_ms": "<float>",
        "query_count": "<int>" }
    ],
    "score_by_intent": [
      { "intent": "<string>", "avg_confidence": "<float>",
        "query_count": "<int>", "top_model": "<string>" }
    ],
    "queries_over_time": [
      { "date": "<YYYY-MM-DD>", "query_count": "<int>" }
    ],
    "ml_vs_rulebased": {
      "ml_won": "<int>",
      "rule_based_won": "<int>"
    }
  },

  "routing_log": [
    {
      "query":       "<string (max 100 chars)>",
      "intent":      "<string>",
      "model":       "<string>",
      "provider":    "<string>",
      "confidence":  "<float>",
      "latency_ms":  "<float>",
      "cost_usd":    "<float>",
      "ml_won":      "<bool>",
      "strategy":    "<string>",
      "complexity":  "<string>",
      "created_at":  "<ISO>"
    }
  ]
}
```
