from fastapi import FastAPI, Body
import sqlite3
from fastapi.middleware.cors import CORSMiddleware

from recommender import (
    train_all_recommenders,
    build_user_item_matrix,
    build_item_user_index,
    recommend_top_n,
    predict_rating,
    pearson_correlation  # make sure this exists
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
# HELPER
# =========================

def build_poster_url(poster_path):
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# =========================
# LOAD TRAINED MODELS
# =========================

models = train_all_recommenders("ml-latest-small")
heuristic_model = models.heuristic
pagerank_model = models.pagerank
node2vec_model = models.node2vec

ratings = heuristic_model.ratings
matrix = heuristic_model.matrix
item_user_index = heuristic_model.item_user_index
all_items = heuristic_model.all_items

# =========================
# 🔥 USER CLUSTERING (CLASSIFICATION)
# =========================

def build_user_clusters(threshold=0.5):
    clusters = []
    visited = set()

    users = list(matrix.keys())

    for user in users:
        if user in visited:
            continue

        cluster = set()
        stack = [user]

        while stack:
            u = stack.pop()

            if u in visited:
                continue

            visited.add(u)
            cluster.add(u)

            for v in users:
                if v != u:
                    sim = pearson_correlation(matrix[u], matrix[v])
                    if sim > threshold:
                        stack.append(v)

        clusters.append(cluster)

    return clusters

clusters = build_user_clusters()

def classify_user(user_id):
    for i, cluster in enumerate(clusters):
        if user_id in cluster:
            return i
    return -1

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
@app.get("/search")
def search_movies(query: str, limit: int = 50, offset: int = 0):
    conn = sqlite3.connect("movies.db")
    cur = conn.cursor()

    cur.execute(
        """
        SELECT movie_id, title, poster_path
        FROM movies
        WHERE LOWER(title) LIKE ?
        ORDER BY title
        LIMIT ? OFFSET ?
        """,
        (f"%{query.lower()}%", limit, offset)
    )

    rows = cur.fetchall()
    conn.close()

    return {
        "results": [
            {
                "movie_id": r[0],
                "title": r[1],
                "poster": build_poster_url(r[2])
            }
            for r in rows
        ],
        "query": query,
        "limit": limit,
        "offset": offset
    }
# =========================
# MOVIES
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
# EXISTING USER RECOMMENDATIONS (UNCHANGED)
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
            "poster": build_poster_url(poster_path),
            "cluster": classify_user(user_id)  # 👈 NEW
        })

    conn.close()

    return {"recommendations": results}

# =========================
# 🔥 NEW USER (CLASSIFICATION BASED)
# =========================

@app.post("/recommend/new")
def recommend_new_user(data: dict = Body(...)):
    return recommend_new_user_heuristic(data)


def _format_selected_movie_results(recommendations):
    conn = sqlite3.connect("movies.db")
    cur = conn.cursor()

    results = []
    for item in recommendations:
        cur.execute(
            "SELECT poster_path FROM movies WHERE movie_id = ?",
            (item["movie_id"],)
        )
        row = cur.fetchone()
        poster_path = row[0] if row else None

        payload = {
            "movie_id": item["movie_id"],
            "title": item["title"],
            "poster": build_poster_url(poster_path),
        }
        payload.update({k: v for k, v in item.items() if k not in {"movie_id", "title"}})
        results.append(payload)

    conn.close()
    return {"recommendations": results}


@app.post("/recommend/new/heuristic")
def recommend_new_user_heuristic(data: dict = Body(...)):
    selected_movies = data.get("movies", [])

    if not selected_movies:
        return {"recommendations": []}

    recommendations = heuristic_model.recommend_for_selected_movies(selected_movies, n=10)
    return _format_selected_movie_results(recommendations)


@app.post("/recommend/new/pagerank")
def recommend_new_user_pagerank(data: dict = Body(...)):
    selected_movies = data.get("movies", [])

    if not selected_movies:
        return {"recommendations": []}

    recommendations = pagerank_model.recommend_for_selected_movies(selected_movies, n=10)
    return _format_selected_movie_results(recommendations)


@app.post("/recommend/new/node2vec")
def recommend_new_user_node2vec(data: dict = Body(...)):
    selected_movies = data.get("movies", [])

    if not selected_movies:
        return {"recommendations": []}

    recommendations = node2vec_model.recommend_for_selected_movies(selected_movies, n=10)
    return _format_selected_movie_results(recommendations)
