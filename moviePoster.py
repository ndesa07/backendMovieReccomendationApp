import sqlite3
import requests
import time

API_KEY = "60c8cba30d2766c792d01a5db8532b57"

def fetch_poster(tmdb_id):
    if not tmdb_id:
        return None

    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}"
        res = requests.get(url, timeout=5)
        data = res.json()
        return data.get("poster_path")
    except:
        return None

def main():
    conn = sqlite3.connect("movies.db")
    cur = conn.cursor()

    # Only update missing posters
    cur.execute("SELECT movie_id, tmdb_id FROM movies WHERE poster_path IS NULL")
    movies = cur.fetchall()

    print(f"Found {len(movies)} movies to update")

    for movie_id, tmdb_id in movies:
        poster_path = fetch_poster(tmdb_id)

        cur.execute(
            "UPDATE movies SET poster_path = ? WHERE movie_id = ?",
            (poster_path, movie_id)
        )

        print(f"✅ Updated movie {movie_id}")

        time.sleep(0.00001)  # avoid rate limit

    conn.commit()
    conn.close()

    print("🎉 DONE")

# 👇 THIS IS CRITICAL
if __name__ == "__main__":
    main()