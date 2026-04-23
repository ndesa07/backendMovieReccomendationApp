from fastapi import FastAPI, Body
import sqlite3
from fastapi.middleware.cors import CORSMiddleware

from recommender import (
    build_user_item_matrix,
    build_item_user_index,
    recommend_top_n,
    predict_rating
)

# =========================
# APP SETUP
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# HELPER (NO API CALLS)
# =========================

def build_poster_url(poster_path):
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# =========================
# LOAD DATA INTO MEMORY
# =========================

def load_from_db():
    conn = sqlite3.connect("movies.db")
    cur = conn.cursor()

    cur.execute("SELECT user_id, movie_id, rating FROM ratings")
    rows = cur.fetchall()

    conn.close()

    return [(u, m, r, 0) for (u, m, r) in rows]

ratings = load_from_db()
matrix = build_user_item_matrix(ratings)
item_user_index = build_item_user_index(matrix)
all_items = list({m for _, m, _, _ in ratings})

# =========================
# USERS
# =========================

@app.get("/users")
def get_users(limit: int = 25, offset: int = 0):
    conn = sqlite3.connect("movies.db")
    cur = conn.cursor()

    cur.execute(
        """
        SELECT DISTINCT user_id 
        FROM ratings 
        ORDER BY user_id 
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    )

    rows = cur.fetchall()
    conn.close()

    return {
        "users": [
            {"id": str(r[0]), "name": f"User {r[0]}"}
            for r in rows
        ],
        "limit": limit,
        "offset": offset,
    }

@app.get("/users/search")
def search_users(query: str, limit: int = 25):
    conn = sqlite3.connect("movies.db")
    cur = conn.cursor()

    cur.execute(
        """
        SELECT DISTINCT user_id 
        FROM ratings
        WHERE CAST(user_id AS TEXT) LIKE ?
        ORDER BY user_id
        LIMIT ?
        """,
        (f"%{query}%", limit),
    )

    rows = cur.fetchall()
    conn.close()

    return {
        "users": [
            {"id": str(r[0]), "name": f"User {r[0]}"}
            for r in rows
        ]
    }

# =========================
# MOVIES (FAST 🚀)
# =========================

@app.get("/movies")
def get_movies(limit: int = 25, offset: int = 0):
    conn = sqlite3.connect("movies.db")
    cur = conn.cursor()

    cur.execute(
        "SELECT movie_id, title, poster_path FROM movies LIMIT ? OFFSET ?",
        (limit, offset)
    )

    rows = cur.fetchall()
    conn.close()

    return {
        "movies": [
            {
                "movie_id": r[0],
                "title": r[1],
                "poster": build_poster_url(r[2])
            }
            for r in rows
        ]
    }

# =========================
# EXISTING USER RECOMMENDATIONS
# =========================

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    recs = recommend_top_n(user_id, matrix, item_user_index, all_items, n=10)

    conn = sqlite3.connect("movies.db")
    cur = conn.cursor()

    results = []

    for movie_id in recs:
        cur.execute(
            "SELECT title, poster_path FROM movies WHERE movie_id = ?",
            (movie_id,)
        )
        row = cur.fetchone()

        title = row[0] if row else "Unknown Movie"
        poster_path = row[1] if row else None

        score = predict_rating(user_id, movie_id, matrix, item_user_index, k=20)

        results.append({
            "movie_id": movie_id,
            "title": title,
            "predicted_rating": round(score, 2),
            "poster": build_poster_url(poster_path)
        })

    conn.close()

    return {"recommendations": results}

# =========================
# NEW USER (COLD START)
# =========================

@app.post("/recommend/new")
def recommend_new_user(data: dict = Body(...)):
    movies = data.get("movies", [])

    if not movies:
        return {"recommendations": []}

    fake_user = 9999
    matrix[fake_user] = {m: 5.0 for m in movies}

    recs = recommend_top_n(fake_user, matrix, item_user_index, all_items, n=10)

    conn = sqlite3.connect("movies.db")
    cur = conn.cursor()

    results = []

    for movie_id in recs:
        cur.execute(
            "SELECT title, poster_path FROM movies WHERE movie_id = ?",
            (movie_id,)
        )
        row = cur.fetchone()

        title = row[0] if row else "Unknown Movie"
        poster_path = row[1] if row else None

        score = predict_rating(fake_user, movie_id, matrix, item_user_index, k=20)

        results.append({
            "movie_id": movie_id,
            "title": title,
            "predicted_rating": round(score, 2),
            "poster": build_poster_url(poster_path)
        })

    conn.close()

    return {"recommendations": results}


