-- Migration 001: Initial multi-tenant schema with RLS
-- Applies to: Supabase (Postgres)
-- Run once against your Supabase project via the SQL editor or supabase db push.

-- -----------------------------------------------------------------------
-- Extensions
-- -----------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()

-- -----------------------------------------------------------------------
-- tenants
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tenants (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT        NOT NULL,
    passkey_hash  TEXT        NOT NULL,
    status        TEXT        NOT NULL DEFAULT 'DRAFT',
    config        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    share_token   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT tenants_status_check CHECK (
        status IN ('DRAFT', 'INGESTED', 'EVALUATED', 'PUBLISHED')
    )
);

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;

-- RLS policy: a DB session may only touch rows whose id matches the
-- session-local setting app.current_tenant_id.
CREATE POLICY tenants_tenant_isolation ON tenants
    USING      (id::text = current_setting('app.current_tenant_id', true))
    WITH CHECK (id::text = current_setting('app.current_tenant_id', true));

-- -----------------------------------------------------------------------
-- eval_logs
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_logs (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  UUID        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    query      TEXT        NOT NULL,
    expected   TEXT        NOT NULL,
    actual     TEXT        NOT NULL,
    score      INTEGER     NOT NULL,
    reason     TEXT        NOT NULL DEFAULT '',
    run_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE eval_logs ENABLE ROW LEVEL SECURITY;

-- RLS policy: tenant isolation via FK to tenants.id
CREATE POLICY eval_logs_tenant_isolation ON eval_logs
    USING      (tenant_id::text = current_setting('app.current_tenant_id', true))
    WITH CHECK (tenant_id::text = current_setting('app.current_tenant_id', true));
