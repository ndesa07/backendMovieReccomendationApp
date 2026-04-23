Here’s a clean **README-style backend documentation** you can use. It’s written in a way that’s good for both **yourself and interviews**.

---

# 📘 Movie Recommendation Backend

## 🚀 Overview

This backend is a **FastAPI-based movie recommendation system** that:

* Serves paginated movie data with posters
* Generates recommendations for users (existing + new)
* Stores movie metadata locally for fast access
* Eliminates external API latency through **precomputed poster caching**

---

## 🏗️ Architecture

```text
Client (React / Next.js)
        ↓
FastAPI Backend
        ↓
SQLite Database (movies.db)
        ↓
Precomputed Poster Data (local)
```

---

## ⚡ Key Features

* ✅ Fast movie pagination (`/movies`)
* ✅ Poster URLs served from local DB (no runtime API calls)
* ✅ Recommendation engine (collaborative filtering)
* ✅ Cold-start recommendations for new users
* ✅ Lightweight SQLite database

---

## 📂 Project Structure

```text
backendMovieRecommendationApp/
│
├── main.py              # FastAPI application
├── recommender.py      # Recommendation logic
├── moviePoster.py      # One-time poster fetching script
├── init_db.py          # Database setup script
├── movies.db           # SQLite database
│
└── ml-latest-small/    # Dataset (MovieLens)
```

---

## 🧠 Data Flow

### 1. Database Initialization

```bash
python init_db.py
```

* Loads:

  * movies
  * ratings
  * tmdb_id mapping

---

### 2. Poster Preprocessing (IMPORTANT)

```bash
python moviePoster.py
```

* Fetches poster paths from TMDB
* Stores `poster_path` in DB
* Runs **once**

👉 This removes runtime API calls (major performance gain)

---

### 3. Backend Startup

```bash
uvicorn main:app --reload
```

Server runs at:

```text
http://127.0.0.1:8000
```

---

## 📡 API Endpoints

---

### 🎬 Get Movies (Paginated)

```http
GET /movies?limit=25&offset=0
```

**Response:**

```json
{
  "movies": [
    {
      "movie_id": 1,
      "title": "Toy Story",
      "poster": "https://image.tmdb.org/t/p/w500/abc.jpg"
    }
  ]
}
```

---

### 👤 Get Users

```http
GET /users
```

---

### 🔍 Search Users

```http
GET /users/search?query=1
```

---

### ⭐ Existing User Recommendations

```http
GET /recommend/{user_id}
```

---

### 🆕 New User Recommendations (Cold Start)

```http
POST /recommend/new
```

**Request:**

```json
{
  "movies": [1, 5, 10]
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "movie_id": 50,
      "title": "Heat",
      "predicted_rating": 4.5,
      "poster": "..."
    }
  ]
}
```

---

## ⚡ Performance Optimisation

### ❌ Before

* API called TMDB per movie
* N+1 request problem
* Slow response (~seconds)

### ✅ After

* Posters precomputed and stored
* No external API calls
* Fast response (~milliseconds)

---

## 🧩 Database Schema

### `movies`

```sql
movie_id INTEGER PRIMARY KEY
title TEXT
tmdb_id INTEGER
poster_path TEXT
```

---

### `ratings`

```sql
user_id INTEGER
movie_id INTEGER
rating REAL
```

---

## 🔁 When to Re-run Scripts

| Script           | When to Run             |
| ---------------- | ----------------------- |
| `init_db.py`     | When rebuilding DB      |
| `moviePoster.py` | After adding new movies |

---

## 🧠 Recommendation Logic

* Collaborative filtering
* User-item matrix
* Similarity-based predictions

---

## 🛠️ Tech Stack

* **FastAPI** — backend framework
* **SQLite** — lightweight database
* **Python** — core logic
* **TMDB API** — poster metadata (one-time use)

---

## 🚀 Future Improvements

* Infinite scroll pagination
* Redis caching
* Model-based recommendations (ML)
* User authentication
* Movie detail endpoints

---

## 💡 Key Engineering Decision

> Posters are pre-fetched and stored locally to eliminate external API calls at runtime.

This avoids:

* latency
* rate limits
* scalability issues

---

## 🏁 Summary

This backend is designed to be:

* Fast ⚡
* Scalable 📈
* Simple 🧩
* Production-ready 💼

