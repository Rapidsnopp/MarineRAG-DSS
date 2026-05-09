"""OAuth authentication helpers for Streamlit."""

from __future__ import annotations

import secrets
from typing import Any

from authlib.integrations.requests_client import OAuth2Session
import requests

from src.config import config


_PROVIDERS = {
    "google": {
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
        "client_id": config.OAUTH_GOOGLE_CLIENT_ID,
        "client_secret": config.OAUTH_GOOGLE_CLIENT_SECRET,
        "scope": "openid email profile",
    },
    "microsoft": {
        "authorize_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "userinfo_url": "https://graph.microsoft.com/oidc/userinfo",
        "client_id": config.OAUTH_MICROSOFT_CLIENT_ID,
        "client_secret": config.OAUTH_MICROSOFT_CLIENT_SECRET,
        "scope": "openid email profile",
    },
}


def _get_provider(provider: str) -> dict[str, str]:
    if provider not in _PROVIDERS:
        raise ValueError("Unsupported provider")
    return _PROVIDERS[provider]


def generate_state(provider: str) -> str:
    token = secrets.token_urlsafe(16)
    return f"{provider}:{token}"


def parse_state(state: str) -> tuple[str, str]:
    if ":" not in state:
        return "", state
    provider, token = state.split(":", 1)
    return provider, token


def get_login_url(provider: str, redirect_uri: str, state: str) -> str:
    cfg = _get_provider(provider)
    session = OAuth2Session(
        cfg["client_id"],
        cfg["client_secret"],
        scope=cfg["scope"],
        redirect_uri=redirect_uri,
    )
    url, _ = session.create_authorization_url(cfg["authorize_url"], state=state)
    return url


def fetch_user_info(provider: str, code: str, redirect_uri: str) -> dict[str, Any]:
    cfg = _get_provider(provider)
    session = OAuth2Session(
        cfg["client_id"],
        cfg["client_secret"],
        scope=cfg["scope"],
        redirect_uri=redirect_uri,
    )
    token = session.fetch_token(cfg["token_url"], code=code, redirect_uri=redirect_uri)
    access_token = token.get("access_token", "")
    resp = requests.get(
        cfg["userinfo_url"],
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()
