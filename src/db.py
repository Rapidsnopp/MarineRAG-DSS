"""MongoDB persistence for users and chat history."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timezone, timedelta
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient

from src.config import config

logger = logging.getLogger(__name__)


_client: MongoClient | None = None
_PASSWORD_ALGO = "pbkdf2_sha256"
_PASSWORD_ITERATIONS = 260000
_TOKEN_BYTES = 32


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_object_id(value: str | ObjectId) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)


def _hash_password(password: str, salt: bytes | None = None, iterations: int | None = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    if iterations is None:
        iterations = _PASSWORD_ITERATIONS
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    hash_b64 = base64.b64encode(derived).decode("ascii")
    return f"{_PASSWORD_ALGO}${iterations}${salt_b64}${hash_b64}"


def _verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    parts = stored_hash.split("$", 3)
    if len(parts) != 4:
        return False
    algo, iter_text, salt_b64, hash_b64 = parts
    if algo != _PASSWORD_ALGO:
        return False
    try:
        iterations = int(iter_text)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
    except (ValueError, binascii.Error):
        return False
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(derived, expected)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _new_token() -> str:
    return secrets.token_urlsafe(_TOKEN_BYTES)


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(config.MONGODB_URI)
    return _client[config.MONGODB_DB]


def ensure_indexes() -> None:
    db = get_db()
    db.users.create_index([("provider", ASCENDING), ("provider_id", ASCENDING)], unique=True)
    db.users.create_index([("email", ASCENDING)])
    db.chat_sessions.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    db.chat_messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])


def get_user_by_email(email: str) -> dict[str, Any] | None:
    db = get_db()
    normalized_email = (email or "").strip().lower()
    if not normalized_email:
        return None
    user = db.users.find_one({"email": normalized_email})
    return _normalize_user(user) if user else None


def create_email_user(email: str, name: str, role: str, password: str) -> dict[str, Any] | None:
    db = get_db()
    normalized_email = (email or "").strip().lower()
    if not normalized_email or not password or len(password) < 8:
        return None
    existing = db.users.find_one({"email": normalized_email})
    if existing:
        return None
    now = _utcnow()
    password_hash = _hash_password(password)
    result = db.users.insert_one(
        {
            "provider": "email",
            "provider_id": normalized_email,
            "email": normalized_email,
            "name": name,
            "role": role,
            "email_verified": False,
            "password_hash": password_hash,
            "created_at": now,
            "last_login": now,
        }
    )
    user = db.users.find_one({"_id": result.inserted_id})
    return _normalize_user(user) if user else None


def authenticate_email_user(email: str, password: str) -> dict[str, Any] | None:
    db = get_db()
    normalized_email = (email or "").strip().lower()
    if not normalized_email or not password:
        return None
    user = db.users.find_one({"email": normalized_email})
    if not user:
        return None
    if user.get("provider") != "email":
        return None
    if "email_verified" in user and not user.get("email_verified", False):
        return None
    if not _verify_password(password, user.get("password_hash", "")):
        return None
    now = _utcnow()
    db.users.update_one({"_id": user["_id"]}, {"$set": {"last_login": now}})
    user["last_login"] = now
    return _normalize_user(user)


def create_email_verification_token(email: str) -> str | None:
    db = get_db()
    normalized_email = (email or "").strip().lower()
    if not normalized_email:
        return None
    user = db.users.find_one({"email": normalized_email, "provider": "email"})
    if not user or user.get("email_verified", False):
        return None
    token = _new_token()
    token_hash = _hash_token(token)
    expires_at = _utcnow() + timedelta(hours=config.EMAIL_VERIFY_TOKEN_HOURS)
    db.users.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "email_verify_token_hash": token_hash,
                "email_verify_expires_at": expires_at,
            }
        },
    )
    return token


def verify_email_token(token: str) -> bool:
    db = get_db()
    if not token:
        return False
    token_hash = _hash_token(token)
    now = _utcnow()
    user = db.users.find_one(
        {
            "email_verify_token_hash": token_hash,
            "email_verify_expires_at": {"$gte": now},
        }
    )
    if not user:
        return False
    db.users.update_one(
        {"_id": user["_id"]},
        {
            "$set": {"email_verified": True},
            "$unset": {"email_verify_token_hash": "", "email_verify_expires_at": ""},
        },
    )
    return True


def create_password_reset_token(email: str) -> str | None:
    db = get_db()
    normalized_email = (email or "").strip().lower()
    if not normalized_email:
        return None
    user = db.users.find_one({"email": normalized_email, "provider": "email"})
    if not user or user.get("email_verified") is False:
        return None
    token = _new_token()
    token_hash = _hash_token(token)
    expires_at = _utcnow() + timedelta(hours=config.PASSWORD_RESET_TOKEN_HOURS)
    db.users.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "password_reset_token_hash": token_hash,
                "password_reset_expires_at": expires_at,
            }
        },
    )
    return token


def reset_password_with_token(token: str, new_password: str) -> bool:
    db = get_db()
    if not token or not new_password or len(new_password) < 8:
        return False
    token_hash = _hash_token(token)
    now = _utcnow()
    user = db.users.find_one(
        {
            "password_reset_token_hash": token_hash,
            "password_reset_expires_at": {"$gte": now},
        }
    )
    if not user:
        return False
    password_hash = _hash_password(new_password)
    db.users.update_one(
        {"_id": user["_id"]},
        {
            "$set": {"password_hash": password_hash},
            "$unset": {
                "password_reset_token_hash": "",
                "password_reset_expires_at": "",
            },
        },
    )
    return True


def upsert_user(provider: str, provider_id: str, email: str, name: str, role: str) -> dict[str, Any]:
    db = get_db()
    now = _utcnow()
    normalized_email = (email or "").strip().lower()

    # If this email already exists (e.g., seeded expert/admin), link it to
    # the OAuth provider on first login so role is preserved.
    if normalized_email:
        existing_by_provider = db.users.find_one(
            {"provider": provider, "provider_id": provider_id}
        )
        if existing_by_provider:
            result = db.users.find_one_and_update(
                {"_id": existing_by_provider["_id"]},
                {
                    "$set": {
                        "email": normalized_email,
                        "name": name,
                        "last_login": now,
                    }
                },
                return_document=True,
            )
            return _normalize_user(result)

        existing_by_email = db.users.find_one({"email": normalized_email})
        if existing_by_email:
            result = db.users.find_one_and_update(
                {"_id": existing_by_email["_id"]},
                {
                    "$set": {
                        "provider": provider,
                        "provider_id": provider_id,
                        "email": normalized_email,
                        "name": name,
                        "last_login": now,
                    }
                },
                return_document=True,
            )
            return _normalize_user(result)

    update = {
        "$set": {
            "email": normalized_email or email,
            "name": name,
            "last_login": now,
            "email_verified": True,
        },
        "$setOnInsert": {
            "provider": provider,
            "provider_id": provider_id,
            "role": role,
            "email_verified": True,
            "created_at": now,
        },
    }
    result = db.users.find_one_and_update(
        {"provider": provider, "provider_id": provider_id},
        update,
        upsert=True,
        return_document=True,
    )
    return _normalize_user(result)


def _normalize_user(user: dict | None) -> dict[str, Any]:
    if not user:
        return {}
    user["id"] = str(user.get("_id"))
    user.pop("_id", None)
    user.pop("password_hash", None)
    user.pop("email_verify_token_hash", None)
    user.pop("email_verify_expires_at", None)
    user.pop("password_reset_token_hash", None)
    user.pop("password_reset_expires_at", None)
    return user


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    db = get_db()
    user = db.users.find_one({"_id": ObjectId(user_id)})
    return _normalize_user(user) if user else None


def list_users() -> list[dict[str, Any]]:
    db = get_db()
    users = []
    for user in db.users.find({}, {"email": 1, "name": 1, "role": 1}).sort("email", ASCENDING):
        users.append(_normalize_user(user))
    return users


def list_user_ids_by_role(role: str | None) -> list[ObjectId]:
    if not role:
        return []
    db = get_db()
    return [user["_id"] for user in db.users.find({"role": role}, {"_id": 1})]


def update_user_role(user_id: str, role: str) -> None:
    db = get_db()
    db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"role": role}})


def create_chat_session(user_id: str, title: str) -> str:
    db = get_db()
    now = _utcnow()
    result = db.chat_sessions.insert_one(
        {
            "user_id": ObjectId(user_id),
            "title": title,
            "created_at": now,
            "updated_at": now,
        }
    )
    return str(result.inserted_id)


def touch_chat_session(session_id: str) -> None:
    db = get_db()
    db.chat_sessions.update_one(
        {"_id": ObjectId(session_id)}, {"$set": {"updated_at": _utcnow()}}
    )


def append_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> None:
    db = get_db()
    db.chat_messages.insert_one(
        {
            "session_id": ObjectId(session_id),
            "user_id": ObjectId(user_id),
            "role": role,
            "content": content,
            "sources": sources or [],
            "created_at": _utcnow(),
        }
    )
    touch_chat_session(session_id)


def list_sessions(user_id: str | None = None) -> list[dict[str, Any]]:
    db = get_db()
    query = {"user_id": ObjectId(user_id)} if user_id else {}
    sessions = []
    for session in db.chat_sessions.find(query).sort("updated_at", DESCENDING):
        session["id"] = str(session.get("_id"))
        session.pop("_id", None)
        session["user_id"] = str(session.get("user_id"))
        sessions.append(session)
    return sessions


def list_sessions_filtered(
    user_ids: list[ObjectId] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    db = get_db()
    query: dict[str, Any] = {}
    if user_ids:
        query["user_id"] = {"$in": user_ids}
    if start or end:
        query["created_at"] = {}
        if start:
            query["created_at"]["$gte"] = start
        if end:
            query["created_at"]["$lte"] = end
    cursor = db.chat_sessions.find(query).sort("updated_at", DESCENDING)
    if limit:
        cursor = cursor.limit(limit)
    sessions = []
    for session in cursor:
        session["id"] = str(session.get("_id"))
        session.pop("_id", None)
        session["user_id"] = str(session.get("user_id"))
        sessions.append(session)
    return sessions


def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    db = get_db()
    messages = []
    for message in db.chat_messages.find({"session_id": ObjectId(session_id)}).sort(
        "created_at", ASCENDING
    ):
        message["id"] = str(message.get("_id"))
        message.pop("_id", None)
        message["session_id"] = str(message.get("session_id"))
        message["user_id"] = str(message.get("user_id"))
        messages.append(message)
    return messages


def list_messages_filtered(
    user_ids: list[ObjectId] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    role: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    db = get_db()
    query: dict[str, Any] = {}
    if user_ids:
        query["user_id"] = {"$in": user_ids}
    if role:
        query["role"] = role
    if start or end:
        query["created_at"] = {}
        if start:
            query["created_at"]["$gte"] = start
        if end:
            query["created_at"]["$lte"] = end
    cursor = db.chat_messages.find(query).sort("created_at", DESCENDING)
    if limit:
        cursor = cursor.limit(limit)
    messages = []
    for message in cursor:
        message["id"] = str(message.get("_id"))
        message.pop("_id", None)
        message["session_id"] = str(message.get("session_id"))
        message["user_id"] = str(message.get("user_id"))
        messages.append(message)
    return messages


def get_admin_stats() -> dict[str, Any]:
    db = get_db()
    return {
        "users": db.users.count_documents({}),
        "sessions": db.chat_sessions.count_documents({}),
        "messages": db.chat_messages.count_documents({}),
    }


def get_admin_analytics(
    user_ids: list[ObjectId] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    db = get_db()
    message_match: dict[str, Any] = {}
    session_match: dict[str, Any] = {}

    if user_ids:
        message_match["user_id"] = {"$in": user_ids}
        session_match["user_id"] = {"$in": user_ids}
    if start or end:
        message_match["created_at"] = {}
        session_match["created_at"] = {}
        if start:
            message_match["created_at"]["$gte"] = start
            session_match["created_at"]["$gte"] = start
        if end:
            message_match["created_at"]["$lte"] = end
            session_match["created_at"]["$lte"] = end

    messages_total = db.chat_messages.count_documents(message_match)
    sessions_total = db.chat_sessions.count_documents(session_match)
    active_users = len(db.chat_messages.distinct("user_id", message_match))

    messages_per_day = list(
        db.chat_messages.aggregate(
            [
                {"$match": message_match} if message_match else {"$match": {}},
                {
                    "$group": {
                        "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"_id": 1}},
            ]
        )
    )

    sessions_per_day = list(
        db.chat_sessions.aggregate(
            [
                {"$match": session_match} if session_match else {"$match": {}},
                {
                    "$group": {
                        "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"_id": 1}},
            ]
        )
    )

    top_users = list(
        db.chat_messages.aggregate(
            [
                {"$match": message_match} if message_match else {"$match": {}},
                {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10},
                {
                    "$lookup": {
                        "from": "users",
                        "localField": "_id",
                        "foreignField": "_id",
                        "as": "user",
                    }
                },
                {"$unwind": {"path": "$user", "preserveNullAndEmptyArrays": True}},
            ]
        )
    )

    role_match = {"_id": {"$in": user_ids}} if user_ids else {}
    role_breakdown = list(
        db.users.aggregate(
            [
                {"$match": role_match} if role_match else {"$match": {}},
                {"$group": {"_id": "$role", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ]
        )
    )

    return {
        "messages_total": messages_total,
        "sessions_total": sessions_total,
        "active_users": active_users,
        "messages_per_day": [
            {"date": item["_id"], "count": item["count"]} for item in messages_per_day
        ],
        "sessions_per_day": [
            {"date": item["_id"], "count": item["count"]} for item in sessions_per_day
        ],
        "top_users": [
            {
                "user_id": str(item.get("_id")),
                "email": item.get("user", {}).get("email", ""),
                "name": item.get("user", {}).get("name", ""),
                "count": item.get("count", 0),
            }
            for item in top_users
        ],
        "role_breakdown": [
            {"role": item.get("_id", "unknown"), "count": item.get("count", 0)}
            for item in role_breakdown
        ],
    }
