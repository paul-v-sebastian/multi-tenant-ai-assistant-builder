from __future__ import annotations

import pytest

from src.tenants import (
    TENANT_STATUS_LIFECYCLE,
    TenantStatus,
    build_tenant_namespace,
    coerce_tenant_status,
    is_valid_tenant_status_transition,
)


def test_tenant_status_lifecycle_matches_poc_spec():
    assert TENANT_STATUS_LIFECYCLE == (
        TenantStatus.DRAFT,
        TenantStatus.INGESTED,
        TenantStatus.EVALUATED,
        TenantStatus.PUBLISHED,
    )


def test_coerce_tenant_status_accepts_enum_and_string():
    assert coerce_tenant_status(TenantStatus.DRAFT) is TenantStatus.DRAFT
    assert coerce_tenant_status("INGESTED") is TenantStatus.INGESTED
    assert coerce_tenant_status(None) is None


def test_coerce_tenant_status_rejects_unknown_values():
    with pytest.raises(ValueError):
        coerce_tenant_status("ARCHIVED")


def test_status_transition_only_allows_linear_progression():
    assert is_valid_tenant_status_transition(None, TenantStatus.DRAFT) is True
    assert is_valid_tenant_status_transition(TenantStatus.DRAFT, TenantStatus.INGESTED) is True
    assert is_valid_tenant_status_transition(TenantStatus.INGESTED, TenantStatus.EVALUATED) is True
    assert is_valid_tenant_status_transition(TenantStatus.EVALUATED, TenantStatus.PUBLISHED) is True
    assert is_valid_tenant_status_transition(TenantStatus.DRAFT, TenantStatus.PUBLISHED) is False
    assert is_valid_tenant_status_transition(TenantStatus.PUBLISHED, TenantStatus.DRAFT) is False


def test_build_tenant_namespace_prefers_tenant_id():
    assert build_tenant_namespace("tenant-123", "legacy-namespace") == "tenant-123"
    assert build_tenant_namespace(None, "legacy-namespace") == "legacy-namespace"
