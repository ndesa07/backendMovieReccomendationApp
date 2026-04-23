import sqlite3
import csv

conn = sqlite3.connect("movies.db")
cur = conn.cursor()

# =========================
# CREATE TABLES (UPDATED)
# =========================

cur.execute("""
CREATE TABLE IF NOT EXISTS ratings (
    user_id INTEGER,
    movie_id INTEGER,
    rating REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS movies (
    movie_id INTEGER PRIMARY KEY,
    title TEXT,
    tmdb_id INTEGER,
    poster_path TEXT   -- ✅ NEW COLUMN
)
""")

# =========================
# LOAD LINKS FIRST (important)
# =========================

tmdb_map = {}

with open("ml-latest-small/links.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        movie_id = int(row["movieId"])
        tmdb_id = row["tmdbId"]

        if tmdb_id:
            tmdb_map[movie_id] = int(tmdb_id)

# =========================
# LOAD MOVIES (WITH poster_path)
# =========================

with open("ml-latest-small/movies.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        movie_id = int(row["movieId"])
        title = row["title"]

        cur.execute(
            """
            INSERT INTO movies (movie_id, title, tmdb_id, poster_path)
            VALUES (?, ?, ?, ?)
            """,
            (
                movie_id,
                title,
                tmdb_map.get(movie_id),
                None  # ✅ initially empty, script will populate
            )
        )

# =========================
# LOAD RATINGS
# =========================

with open("ml-latest-small/ratings.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cur.execute(
            "INSERT INTO ratings VALUES (?, ?, ?)",
            (
                int(row["userId"]),
                int(row["movieId"]),
                float(row["rating"])
            )
        )

# =========================
# FINALIZE
# =========================

conn.commit()
conn.close()

print("✅ Database created with tmdb_id + poster_path successfully!")