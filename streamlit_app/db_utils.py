import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_PARAMS = {
  'dbname': os.getenv('POSTGRES_DB', 'postgres'),
  'user': os.getenv('POSTGRES_USER', 'postgres'),
  'password': os.getenv('POSTGRES_PASSWORD', 'password'),
  'host': os.getenv('POSTGRES_HOST', 'localhost'),
  'port': os.getenv('POSTGRES_PORT', '5432'),
}

def connect_to_database():
  try:
      conn = psycopg2.connect(**DB_PARAMS)
      return conn
  except Exception as e:
      print(f"PostgreSQL connection error: {e}")
      return None

def ensure_database_setup():
  try:
      conn = connect_to_database()
      if not conn:
          return False
      cursor = conn.cursor()
      cursor.execute("""
          CREATE TABLE IF NOT EXISTS history (
              id SERIAL PRIMARY KEY,
              image_path TEXT,
              caption TEXT
          )
      """)
      conn.commit()
      cursor.close()
      conn.close()
      return True
  except Exception as e:
      print(f"Database setup error: {e}")
      return False

def save_caption(image_path, caption):
  conn = connect_to_database()
  if not conn:
      return
  cursor = conn.cursor()
  cursor.execute("INSERT INTO history (image_path, caption) VALUES (%s, %s)", (image_path, caption))
  conn.commit()
  cursor.close()
  conn.close()

def get_recent_captions(limit=5):
  conn = connect_to_database()
  if not conn:
      return []
  cursor = conn.cursor()
  cursor.execute("SELECT image_path, caption FROM history ORDER BY id DESC LIMIT %s", (limit,))
  rows = cursor.fetchall()
  cursor.close()
  conn.close()
  return rows
