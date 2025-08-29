import os
import sqlite3
import psycopg2
from dotenv import load_dotenv

load_dotenv()
database_url = os.getenv('DATABASE_URL')

try:
    if database_url.startswith('sqlite'):
        conn = sqlite3.connect(database_url.replace('sqlite:///', ''))
        print("Successfully connected to SQLite database!")
        conn.close()
    else:
        conn = psycopg2.connect(database_url)
        print("Successfully connected to PostgreSQL database!")
        conn.close()
except Exception as e:
    print(f"Failed to connect to database: {str(e)}")