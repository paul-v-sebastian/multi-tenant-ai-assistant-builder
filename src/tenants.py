from __future__ import annotations

from enum import StrEnum


class TenantStatus(StrEnum):
    DRAFT = "DRAFT"
    INGESTED = "INGESTED"
    EVALUATED = "EVALUATED"
    PUBLISHED = "PUBLISHED"


TENANT_STATUS_LIFECYCLE: tuple[TenantStatus, ...] = (
    TenantStatus.DRAFT,
    TenantStatus.INGESTED,
    TenantStatus.EVALUATED,
    TenantStatus.PUBLISHED,
)


def coerce_tenant_status(value: str | TenantStatus | None) -> TenantStatus | None:
    if value is None or value == "":
        return None
    if isinstance(value, TenantStatus):
        return value
    return TenantStatus(value)


def is_valid_tenant_status_transition(
    current_status: str | TenantStatus | None,
    next_status: str | TenantStatus,
) -> bool:
    current = coerce_tenant_status(current_status)
    target = coerce_tenant_status(next_status)
    if target is None:
        return False
    if current is None:
        return target is TenantStatus.DRAFT
    if current is target:
        return True

    current_index = TENANT_STATUS_LIFECYCLE.index(current)
    target_index = TENANT_STATUS_LIFECYCLE.index(target)
    return target_index == current_index + 1


def build_tenant_namespace(tenant_id: str | None, fallback_namespace: str | None = None) -> str | None:
    tenant_value = (tenant_id or "").strip()
    if tenant_value:
        return tenant_value
    return fallback_namespace
