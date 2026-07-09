CREATE TABLE IF NOT EXISTS routing_responses (
    id BIGSERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    selected_model TEXT NOT NULL,
    strategy TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    response_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_routing_responses_created_at
    ON routing_responses(created_at DESC);
