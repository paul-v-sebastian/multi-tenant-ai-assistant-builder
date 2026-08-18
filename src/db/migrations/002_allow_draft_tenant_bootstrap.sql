-- Migration 002: Allow bootstrap creation of new draft tenants
-- Applies to: Supabase (Postgres)
-- Run after 001_initial_schema.sql.

-- Keep the strict tenant-isolation policy for tenant-scoped reads/writes, but
-- permit unauthenticated bootstrap INSERTs for brand-new DRAFT tenants so the
-- app can create the first row before tenant context exists.
CREATE POLICY tenants_bootstrap_insert ON tenants
    FOR INSERT
    WITH CHECK (
        status = 'DRAFT'
        AND NULLIF(BTRIM(name), '') IS NOT NULL
        AND NULLIF(BTRIM(passkey_hash), '') IS NOT NULL
        AND share_token IS NULL
    );
