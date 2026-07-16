CREATE TABLE IF NOT EXISTS routing_responses (
    id BIGSERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    ml_router_predicted TEXT,
    rule_based_would_pick TEXT,
    default_model_used TEXT NOT NULL,
    provider TEXT NOT NULL,
    note TEXT,
    intent TEXT NOT NULL,
    complexity TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    tokens INTEGER DEFAULT 0,
    strategy TEXT NOT NULL,
    response_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_routing_responses_created_at
    ON routing_responses(created_at DESC);
