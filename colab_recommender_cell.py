from collections import Counter, defaultdict
import math
import re

import pandas as pd


# ------------------------------------------------------------
# NOTEBOOK SETUP
# ------------------------------------------------------------
# This cell assumes you already ran:
#
# ratings = pd.read_csv('ml-latest-small/ratings.csv')
# ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)
#
# movies_df = pd.read_csv('ml-latest-small/movies.csv')
# movies_df.rename(columns={'movieId': 'movie_id'}, inplace=True)
#
# tags_df = pd.read_csv('ml-latest-small/tags.csv')
# tags_df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)
#
# Then paste this cell and run:
# model = train_recommender_from_dataframes(ratings, movies_df, tags_df)
# model.recommend_for_selected_movies([1, 296, 593], n=10)


TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_token(token):
    return str(token).strip().lower().replace(" ", "_")


def tokenize_text(text):
    return [match.group(0) for match in TOKEN_RE.finditer(str(text).lower())]


def prepare_movies(movies_df):
    movies = {}
    for row in movies_df.itertuples(index=False):
        movie_id = int(row.movie_id)
        genres_raw = getattr(row, "genres", "")
        genres = []
        if pd.notna(genres_raw) and genres_raw != "(no genres listed)":
            genres = [normalize_token(part) for part in str(genres_raw).split("|")]
        movies[movie_id] = {
            "title": row.title,
            "genres": genres,
        }
    return movies


def prepare_ratings(ratings_df):
    ratings = []
    for row in ratings_df.itertuples(index=False):
        timestamp = int(row.timestamp) if hasattr(row, "timestamp") and pd.notna(row.timestamp) else 0
        ratings.append(
            (
                int(row.user_id),
                int(row.movie_id),
                float(row.rating),
                timestamp,
            )
        )
    return ratings


def prepare_tags(tags_df):
    tags_by_movie = defaultdict(Counter)
    for row in tags_df.itertuples(index=False):
        if not hasattr(row, "tag") or pd.isna(row.tag):
            continue
        movie_id = int(row.movie_id)
        for token in tokenize_text(row.tag):
            tags_by_movie[movie_id][token] += 1
    return dict(tags_by_movie)


def build_user_item_matrix(ratings):
    matrix = {}
    for user_id, item_id, rating, _ in ratings:
        if user_id not in matrix:
            matrix[user_id] = {}
        matrix[user_id][item_id] = rating
    return matrix


def build_item_user_index(matrix):
    index = {}
    for user_id, items in matrix.items():
        for item_id in items:
            if item_id not in index:
                index[item_id] = set()
            index[item_id].add(user_id)
    return index


def user_mean(user_ratings):
    if not user_ratings:
        return 0.0
    return sum(user_ratings.values()) / len(user_ratings)


def pearson_correlation(ratings_a, ratings_b):
    common = [item_id for item_id in ratings_a if item_id in ratings_b]
    count = len(common)
    if count < 2:
        return 0.0

    mean_a = sum(ratings_a[item_id] for item_id in common) / count
    mean_b = sum(ratings_b[item_id] for item_id in common) / count

    numerator = 0.0
    sq_dev_a = 0.0
    sq_dev_b = 0.0
    for item_id in common:
        dev_a = ratings_a[item_id] - mean_a
        dev_b = ratings_b[item_id] - mean_b
        numerator += dev_a * dev_b
        sq_dev_a += dev_a * dev_a
        sq_dev_b += dev_b * dev_b

    denominator = math.sqrt(sq_dev_a) * math.sqrt(sq_dev_b)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def predict_rating(user_id, item_id, matrix, item_user_index, k=20, min_support=2):
    if user_id not in matrix:
        return 3.0

    user_ratings = matrix[user_id]
    u_mean = user_mean(user_ratings)
    candidates = item_user_index.get(item_id, set())

    neighbours = []
    for other_id in candidates:
        if other_id == user_id:
            continue
        other_ratings = matrix[other_id]
        sim = pearson_correlation(user_ratings, other_ratings)
        overlap = sum(1 for rated_item in user_ratings if rated_item in other_ratings)
        if sim > 0 and overlap >= min_support:
            neighbours.append((sim, other_id))

    if not neighbours:
        return u_mean

    neighbours.sort(reverse=True)
    top_k = neighbours[:k]

    numerator = 0.0
    denominator = 0.0
    for sim, other_id in top_k:
        other_ratings = matrix[other_id]
        v_mean = user_mean(other_ratings)
        numerator += sim * (other_ratings[item_id] - v_mean)
        denominator += abs(sim)

    if denominator == 0.0:
        return u_mean

    prediction = u_mean + numerator / denominator
    return max(1.0, min(5.0, prediction))


def recommend_top_n(user_id, matrix, item_user_index, all_items, n=10):
    if user_id not in matrix:
        return []

    rated = set(matrix[user_id].keys())
    predictions = []
    for item_id in all_items:
        if item_id in rated:
            continue
        pred = predict_rating(user_id, item_id, matrix, item_user_index, k=20)
        predictions.append((pred, item_id))

    predictions.sort(reverse=True)
    return [item_id for _, item_id in predictions[:n]]


def cosine_similarity(vec_a, norm_a, vec_b, norm_b):
    if not vec_a or not vec_b or norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
        norm_a, norm_b = norm_b, norm_a

    dot = 0.0
    for key, value in vec_a.items():
        dot += value * vec_b.get(key, 0.0)
    return dot / (norm_a * norm_b) if dot else 0.0


class RecommenderModel:
    def __init__(
        self,
        movies,
        ratings,
        tags_by_movie,
        matrix,
        item_user_index,
        all_items,
        movie_vectors,
        movie_vector_norms,
        movie_avg_ratings,
        movie_rating_counts,
        liked_by_movie,
        global_mean_rating,
    ):
        self.movies = movies
        self.ratings = ratings
        self.tags_by_movie = tags_by_movie
        self.matrix = matrix
        self.item_user_index = item_user_index
        self.all_items = all_items
        self.movie_vectors = movie_vectors
        self.movie_vector_norms = movie_vector_norms
        self.movie_avg_ratings = movie_avg_ratings
        self.movie_rating_counts = movie_rating_counts
        self.liked_by_movie = liked_by_movie
        self.global_mean_rating = global_mean_rating

    def _movie_title(self, movie_id):
        movie = self.movies.get(movie_id, {})
        return movie.get("title", f"Movie {movie_id}")

    def _build_profile_vector(self, selected_movie_ids):
        profile = defaultdict(float)
        selected = [movie_id for movie_id in selected_movie_ids if movie_id in self.movie_vectors]
        if not selected:
            return {}, 0.0

        weight = 1.0 / len(selected)
        for movie_id in selected:
            for feature, value in self.movie_vectors[movie_id].items():
                profile[feature] += value * weight

        norm = math.sqrt(sum(value * value for value in profile.values()))
        return dict(profile), norm

    def _collaborative_item_similarity(self, movie_a, movie_b):
        users_a = self.liked_by_movie.get(movie_a, set())
        users_b = self.liked_by_movie.get(movie_b, set())
        if not users_a or not users_b:
            return 0.0
        return len(users_a & users_b) / math.sqrt(len(users_a) * len(users_b))

    def _reason_tokens(self, candidate_id, selected_movie_ids, limit=3):
        candidate_vector = self.movie_vectors.get(candidate_id, {})
        if not candidate_vector:
            return []

        profile, _ = self._build_profile_vector(selected_movie_ids)
        shared = []
        for feature, weight in candidate_vector.items():
            if feature in profile:
                shared.append((weight * profile[feature], feature))

        shared.sort(reverse=True)

        reasons = []
        for _, feature in shared:
            label, raw = feature.split(":", 1)
            pretty = raw.replace("_", " ")
            if label == "genre":
                reason = f"shares the {pretty} genre"
            else:
                reason = f"matches the tag '{pretty}'"
            if reason not in reasons:
                reasons.append(reason)
            if len(reasons) == limit:
                break
        return reasons

    def _score_candidate(self, candidate_id, selected_movie_ids, profile_vector, profile_norm):
        candidate_vector = self.movie_vectors.get(candidate_id, {})
        candidate_norm = self.movie_vector_norms.get(candidate_id, 0.0)
        content_score = cosine_similarity(profile_vector, profile_norm, candidate_vector, candidate_norm)

        collaborative_scores = [
            self._collaborative_item_similarity(selected_id, candidate_id)
            for selected_id in selected_movie_ids
            if selected_id != candidate_id
        ]
        collaborative_score = (
            sum(collaborative_scores) / len(collaborative_scores)
            if collaborative_scores
            else 0.0
        )

        avg_rating = self.movie_avg_ratings.get(candidate_id, self.global_mean_rating)
        rating_score = max(0.0, min(1.0, (avg_rating - 3.0) / 2.0))

        max_count = max(self.movie_rating_counts.values()) if self.movie_rating_counts else 1
        popularity = self.movie_rating_counts.get(candidate_id, 0)
        popularity_score = math.log1p(popularity) / math.log1p(max_count) if max_count > 1 else 0.0

        final_score = (
            0.55 * content_score
            + 0.25 * collaborative_score
            + 0.15 * rating_score
            + 0.05 * popularity_score
        )

        return {
            "content_score": content_score,
            "collaborative_score": collaborative_score,
            "predicted_rating": avg_rating,
            "score": final_score,
        }

    def recommend_for_selected_movies(self, selected_movie_ids, n=10):
        selected = [movie_id for movie_id in selected_movie_ids if movie_id in self.movies]
        if not selected:
            return []

        profile_vector, profile_norm = self._build_profile_vector(selected)

        scored = []
        for candidate_id in self.all_items:
            if candidate_id in selected:
                continue

            scores = self._score_candidate(candidate_id, selected, profile_vector, profile_norm)
            if scores["score"] <= 0.0:
                continue

            scored.append(
                {
                    "movie_id": candidate_id,
                    "title": self._movie_title(candidate_id),
                    "score": round(scores["score"], 4),
                    "predicted_rating": round(scores["predicted_rating"], 2),
                    "content_score": round(scores["content_score"], 4),
                    "collaborative_score": round(scores["collaborative_score"], 4),
                    "reason": self._reason_tokens(candidate_id, selected),
                }
            )

        scored.sort(key=lambda item: (item["score"], item["predicted_rating"]), reverse=True)
        return scored[:n]


def build_movie_vectors(movies, tags_by_movie):
    raw_features = {}
    document_frequency = Counter()

    for movie_id, movie in movies.items():
        features = Counter()

        for genre in movie["genres"]:
            features[f"genre:{genre}"] += 2.0

        for token, count in tags_by_movie.get(movie_id, {}).items():
            features[f"tag:{token}"] += min(3.0, 1.0 + math.log1p(count))

        raw_features[movie_id] = dict(features)
        for feature in features:
            document_frequency[feature] += 1

    total_movies = max(1, len(movies))
    vectors = {}
    norms = {}
    for movie_id, features in raw_features.items():
        vector = {}
        for feature, tf in features.items():
            idf = math.log((1 + total_movies) / (1 + document_frequency[feature])) + 1.0
            vector[feature] = tf * idf
        vectors[movie_id] = vector
        norms[movie_id] = math.sqrt(sum(value * value for value in vector.values()))

    return vectors, norms


def compute_movie_rating_stats(ratings):
    totals = defaultdict(float)
    counts = defaultdict(int)
    liked_by_movie = defaultdict(set)

    for user_id, movie_id, rating, _ in ratings:
        totals[movie_id] += rating
        counts[movie_id] += 1
        if rating >= 4.0:
            liked_by_movie[movie_id].add(user_id)

    averages = {
        movie_id: totals[movie_id] / counts[movie_id]
        for movie_id in counts
    }
    return averages, dict(counts), {movie_id: set(users) for movie_id, users in liked_by_movie.items()}


def train_recommender_from_dataframes(ratings_df, movies_df, tags_df):
    movies = prepare_movies(movies_df)
    ratings = prepare_ratings(ratings_df)
    tags_by_movie = prepare_tags(tags_df)

    matrix = build_user_item_matrix(ratings)
    item_user_index = build_item_user_index(matrix)
    movie_vectors, movie_vector_norms = build_movie_vectors(movies, tags_by_movie)
    movie_avg_ratings, movie_rating_counts, liked_by_movie = compute_movie_rating_stats(ratings)
    global_mean_rating = sum(rating for _, _, rating, _ in ratings) / len(ratings) if ratings else 3.0

    model = RecommenderModel(
        movies=movies,
        ratings=ratings,
        tags_by_movie=tags_by_movie,
        matrix=matrix,
        item_user_index=item_user_index,
        all_items=sorted(movies.keys()),
        movie_vectors=movie_vectors,
        movie_vector_norms=movie_vector_norms,
        movie_avg_ratings=movie_avg_ratings,
        movie_rating_counts=movie_rating_counts,
        liked_by_movie=liked_by_movie,
        global_mean_rating=global_mean_rating,
    )

    print(
        f"Model training complete: {len(model.movies)} movies, "
        f"{len(model.ratings)} ratings, {len(model.tags_by_movie)} tagged movies loaded."
    )
    return model
