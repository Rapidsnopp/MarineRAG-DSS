"""Role permissions for MarineRAG-DSS."""

from __future__ import annotations

ROLES = ["citizen", "expert", "manager", "admin"]
NON_ADMIN_ROLES = [role for role in ROLES if role != "admin"]

ROLE_LABELS = {
    "citizen": "Nguoi dan",
    "expert": "Chuyen gia",
    "manager": "Can bo quan ly",
    "admin": "Admin",
}

ROLE_PERMISSIONS = {
    "citizen": {
        "can_view_sources": False,
        "can_view_all_sessions": False,
        "can_view_admin": False,
    },
    "expert": {
        "can_view_sources": True,
        "can_view_all_sessions": False,
        "can_view_admin": False,
    },
    "manager": {
        "can_view_sources": True,
        "can_view_all_sessions": True,
        "can_view_admin": False,
    },
    "admin": {
        "can_view_sources": True,
        "can_view_all_sessions": True,
        "can_view_admin": True,
    },
}


_ROLE_ALIASES = {
    "nguoi dan": "citizen",
    "người dân": "citizen",
    "chuyen gia": "expert",
    "chuyên gia": "expert",
    "can bo quan ly": "manager",
    "cán bộ quản lý": "manager",
}


def normalize_role(role: str | None) -> str:
    normalized = (role or "").strip().lower()
    if not normalized:
        return "citizen"
    return _ROLE_ALIASES.get(normalized, normalized)


def get_permissions(role: str | None) -> dict:
    return ROLE_PERMISSIONS.get(normalize_role(role), ROLE_PERMISSIONS["citizen"])
