from pathlib import Path


def test_bootstrap_tenant_migration_allows_only_draft_insert_policy():
    migration = Path(
        "/home/runner/work/multi-tenant-ai-assistant-builder/"
        "multi-tenant-ai-assistant-builder/src/db/migrations/"
        "002_allow_draft_tenant_bootstrap.sql"
    ).read_text()

    assert "CREATE POLICY tenants_bootstrap_insert ON tenants" in migration
    assert "FOR INSERT" in migration
    assert "status = 'DRAFT'" in migration
    assert "share_token IS NULL" in migration
