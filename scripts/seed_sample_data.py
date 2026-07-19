#!/usr/bin/env python3
# WHY THIS FILE EXISTS:
# Creates sample users and prints their JWT tokens for local development testing.
# Run via: make seed
# Outputs tokens you can paste directly into curl/Postman.

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging
setup_logging("INFO")

from app.core.security import create_access_token, hash_password
from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    print("\n" + "="*60)
    print("  ai_gen_platform — Sample Tokens for Local Development")
    print("="*60)

    users = [
        {"username": "alice", "password": "secret", "role": "admin"},
        {"username": "bob", "password": "secret", "role": "user"},
    ]

    for user in users:
        token = create_access_token(
            subject=user["username"],
            extra_claims={"role": user["role"]},
        )
        print(f"\n  User:  {user['username']} ({user['role']})")
        print(f"  Pass:  {user['password']}")
        print(f"  Token: {token[:40]}...{token[-10:]}")
        print(f"\n  Full token (use in Authorization header):")
        print(f"  Bearer {token}")

    print("\n" + "="*60)
    print(f"  API Key: {settings.api_key}")
    print("  (Use as X-API-Key header for RAG and ingest endpoints)")
    print("="*60 + "\n")
    print("  Hash a new password with:")
    print("  python -c \"from passlib.hash import bcrypt; print(bcrypt.hash('yourpassword'))\"")
    print()


if __name__ == "__main__":
    main()
