#!/usr/bin/env python3
import sqlite3
import bcrypt
import uuid
import random
import string
import json
from datetime import datetime

# Generate API key
random_part = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
api_key = f"cgk_{random_part}"
print(f"Generated API key: {api_key}")

# Create hash
key_hash = bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Key prefix (first 12 chars)
key_prefix = api_key[:12]

# Connect to database
conn = sqlite3.connect('corpgrid.db')
cursor = conn.cursor()

# Get admin user ID
cursor.execute("SELECT id FROM users WHERE username = 'admin'")
user_id = cursor.fetchone()[0]

# Generate UUID for the key
key_id = str(uuid.uuid4())

# Insert API key
now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
scopes = json.dumps(["*"])  # Full access

cursor.execute("""
    INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name, scopes, is_active, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (key_id, user_id, key_hash, key_prefix, "Test API Key", scopes, 1, now))

conn.commit()
conn.close()

print(f"API key created successfully!")
print(f"User ID: {user_id}")
print(f"Key ID: {key_id}")
print(f"\nSave this API key, it won't be shown again:")
print(f"{api_key}")
