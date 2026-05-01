from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import math
import random
import re

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_token(token):
    return token.strip().lower().replace(" ", "_")


def tokenize_text(text):
    return [match.group(0) for match in TOKEN_RE.finditer(text.lower())]


def load_ratings(filepath):
    ratings = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings.append(
                (
                    int(row["userId"]),
                    int(row["movieId"]),
                    float(row["rating"]),
                    int(row["timestamp"]),
                )
            )
    return ratings


def load_movies(filepath):
    movies = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = int(row["movieId"])
            genres = []
            if row["genres"] and row["genres"] != "(no genres listed)":
                genres = [normalize_token(part) for part in row["genres"].split("|")]
            movies[movie_id] = {
                "title": row["title"],
                "genres": genres,
            }
    return movies


def load_tags(filepath):
    tags_by_movie = defaultdict(Counter)
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = int(row["movieId"])
            for token in tokenize_text(row["tag"]):
                tags_by_movie[movie_id][token] += 1
    return dict(tags_by_movie)


def split_by_timestamp(ratings, test_ratio=0.2):
    sorted_ratings = sorted(ratings, key=lambda x: x[3])
    split_idx = int(len(sorted_ratings) * (1.0 - test_ratio))
    train = sorted_ratings[:split_idx]
    test = sorted_ratings[split_idx:]
    return train, test


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


def compute_rmsd(test_data, matrix, item_user_index, k=20):
    squared_errors = []
    for user_id, item_id, actual, _ in test_data:
        predicted = predict_rating(user_id, item_id, matrix, item_user_index, k=k)
        squared_errors.append((predicted - actual) ** 2)

    if not squared_errors:
        return 0.0
    return math.sqrt(sum(squared_errors) / len(squared_errors))


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


def compute_recall_at_n(test_data, matrix, item_user_index, all_items, n=10, threshold=4.0):
    user_relevant = {}
    for user_id, item_id, rating, _ in test_data:
        if rating >= threshold:
            if user_id not in user_relevant:
                user_relevant[user_id] = set()
            user_relevant[user_id].add(item_id)

    recalls = []
    for user_id, relevant in user_relevant.items():
        if not relevant:
            continue
        recommended = set(recommend_top_n(user_id, matrix, item_user_index, all_items, n=n))
        recalls.append(len(recommended & relevant) / len(relevant))

    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


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


def build_interaction_graph(ratings, positive_threshold=4.0):
    graph = nx.Graph()
    for user_id, movie_id, rating, _ in ratings:
        if rating < positive_threshold:
            continue
        user_node = f"u{user_id}"
        movie_node = f"m{movie_id}"
        graph.add_node(user_node, node_type="user", raw_id=user_id)
        graph.add_node(movie_node, node_type="movie", raw_id=movie_id)
        graph.add_edge(user_node, movie_node, weight=rating)
    return graph


def movie_rating_prior(movie_id, movie_avg_ratings, movie_rating_counts, global_mean_rating):
    avg_rating = movie_avg_ratings.get(movie_id, global_mean_rating)
    rating_score = max(0.0, min(1.0, (avg_rating - 3.0) / 2.0))
    max_count = max(movie_rating_counts.values()) if movie_rating_counts else 1
    popularity = movie_rating_counts.get(movie_id, 0)
    popularity_score = math.log1p(popularity) / math.log1p(max_count) if max_count > 1 else 0.0
    return avg_rating, rating_score, popularity_score


@dataclass
class HeuristicRecommenderModel:
    movies: dict
    ratings: list
    tags_by_movie: dict
    matrix: dict
    item_user_index: dict
    all_items: list
    movie_vectors: dict
    movie_vector_norms: dict
    movie_avg_ratings: dict
    movie_rating_counts: dict
    liked_by_movie: dict
    global_mean_rating: float

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

        avg_rating, rating_score, popularity_score = movie_rating_prior(
            candidate_id,
            self.movie_avg_ratings,
            self.movie_rating_counts,
            self.global_mean_rating,
        )

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
                    "model_type": "heuristic",
                }
            )

        scored.sort(key=lambda item: (item["score"], item["predicted_rating"]), reverse=True)
        return scored[:n]


@dataclass
class PageRankRecommenderModel:
    movies: dict
    graph: nx.Graph
    all_items: list
    movie_avg_ratings: dict
    movie_rating_counts: dict
    global_mean_rating: float

    def recommend_for_selected_movies(self, selected_movie_ids, n=10):
        selected = [movie_id for movie_id in selected_movie_ids if movie_id in self.movies]
        if not selected:
            return []

        personalization = {node: 0.0 for node in self.graph.nodes}
        selected_nodes = []
        for movie_id in selected:
            node = f"m{movie_id}"
            if node in personalization:
                personalization[node] = 1.0
                selected_nodes.append(node)

        if not selected_nodes:
            return []

        total = sum(personalization.values())
        personalization = {
            node: value / total
            for node, value in personalization.items()
            if value > 0.0
        }

        scores = nx.pagerank(self.graph, alpha=0.85, personalization=personalization)
        ranked = []

        for movie_id in self.all_items:
            if movie_id in selected:
                continue

            movie_node = f"m{movie_id}"
            pagerank_score = scores.get(movie_node, 0.0)
            if pagerank_score <= 0.0:
                continue

            avg_rating, rating_score, popularity_score = movie_rating_prior(
                movie_id,
                self.movie_avg_ratings,
                self.movie_rating_counts,
                self.global_mean_rating,
            )

            final_score = 0.8 * pagerank_score + 0.15 * rating_score + 0.05 * popularity_score
            ranked.append(
                {
                    "movie_id": movie_id,
                    "title": self.movies[movie_id]["title"],
                    "score": round(final_score, 4),
                    "predicted_rating": round(avg_rating, 2),
                    "pagerank_score": round(pagerank_score, 6),
                    "reason": ["connected to users with similar highly rated movies"],
                    "model_type": "pagerank",
                }
            )

        ranked.sort(key=lambda item: (item["score"], item["predicted_rating"]), reverse=True)
        return ranked[:n]


class Node2VecSkipGram(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.context_embeddings = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)

    def forward(self, center_nodes, context_nodes, negative_nodes):
        center_emb = self.center_embeddings(center_nodes)
        context_emb = self.context_embeddings(context_nodes)
        negative_emb = self.context_embeddings(negative_nodes)

        positive_score = torch.sum(center_emb * context_emb, dim=1)
        positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-9)

        negative_score = torch.bmm(negative_emb, center_emb.unsqueeze(2)).squeeze(2)
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_score) + 1e-9), dim=1)

        return torch.mean(positive_loss + negative_loss)


@dataclass
class Node2VecRecommenderModel:
    movies: dict
    all_items: list
    embeddings: dict
    movie_avg_ratings: dict
    movie_rating_counts: dict
    global_mean_rating: float

    def recommend_for_selected_movies(self, selected_movie_ids, n=10):
        selected = [movie_id for movie_id in selected_movie_ids if movie_id in self.movies and movie_id in self.embeddings]
        if not selected:
            return []

        selected_vectors = np.array([self.embeddings[movie_id] for movie_id in selected])
        profile = np.mean(selected_vectors, axis=0)
        profile_norm = np.linalg.norm(profile)
        if profile_norm == 0.0:
            return []

        ranked = []
        for movie_id in self.all_items:
            if movie_id in selected or movie_id not in self.embeddings:
                continue

            movie_vector = self.embeddings[movie_id]
            denom = profile_norm * np.linalg.norm(movie_vector)
            if denom == 0.0:
                continue
            embedding_score = float(np.dot(profile, movie_vector) / denom)

            avg_rating, rating_score, popularity_score = movie_rating_prior(
                movie_id,
                self.movie_avg_ratings,
                self.movie_rating_counts,
                self.global_mean_rating,
            )

            final_score = 0.75 * embedding_score + 0.2 * rating_score + 0.05 * popularity_score
            ranked.append(
                {
                    "movie_id": movie_id,
                    "title": self.movies[movie_id]["title"],
                    "score": round(final_score, 4),
                    "predicted_rating": round(avg_rating, 2),
                    "embedding_score": round(embedding_score, 4),
                    "reason": ["close in the learned user-movie graph embedding space"],
                    "model_type": "node2vec",
                }
            )

        ranked.sort(key=lambda item: (item["score"], item["predicted_rating"]), reverse=True)
        return ranked[:n]


@dataclass
class RecommenderSuite:
    heuristic: HeuristicRecommenderModel
    pagerank: PageRankRecommenderModel
    node2vec: Node2VecRecommenderModel


def train_heuristic_recommender(data_dir="ml-latest-small"):
    movies = load_movies(f"{data_dir}/movies.csv")
    ratings = load_ratings(f"{data_dir}/ratings.csv")
    tags_by_movie = load_tags(f"{data_dir}/tags.csv")

    matrix = build_user_item_matrix(ratings)
    item_user_index = build_item_user_index(matrix)
    movie_vectors, movie_vector_norms = build_movie_vectors(movies, tags_by_movie)
    movie_avg_ratings, movie_rating_counts, liked_by_movie = compute_movie_rating_stats(ratings)
    global_mean_rating = sum(rating for _, _, rating, _ in ratings) / len(ratings) if ratings else 3.0

    model = HeuristicRecommenderModel(
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
        f"Heuristic model training complete: {len(model.movies)} movies, "
        f"{len(model.ratings)} ratings, {len(model.tags_by_movie)} tagged movies loaded."
    )
    return model


def train_pagerank_recommender(data_dir="ml-latest-small", positive_threshold=4.0):
    movies = load_movies(f"{data_dir}/movies.csv")
    ratings = load_ratings(f"{data_dir}/ratings.csv")
    movie_avg_ratings, movie_rating_counts, _ = compute_movie_rating_stats(ratings)
    global_mean_rating = sum(rating for _, _, rating, _ in ratings) / len(ratings) if ratings else 3.0
    graph = build_interaction_graph(ratings, positive_threshold=positive_threshold)

    model = PageRankRecommenderModel(
        movies=movies,
        graph=graph,
        all_items=sorted(movies.keys()),
        movie_avg_ratings=movie_avg_ratings,
        movie_rating_counts=movie_rating_counts,
        global_mean_rating=global_mean_rating,
    )

    print(
        f"PageRank model training complete: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} positive-rating edges."
    )
    return model


def generate_node2vec_walk(graph, start_node, walk_length, p, q, rng):
    walk = [start_node]

    while len(walk) < walk_length:
        current = walk[-1]
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break

        if len(walk) == 1:
            walk.append(rng.choice(neighbors))
            continue

        previous = walk[-2]
        probabilities = []
        for neighbor in neighbors:
            if neighbor == previous:
                weight = 1.0 / p
            elif graph.has_edge(previous, neighbor):
                weight = 1.0
            else:
                weight = 1.0 / q
            probabilities.append(weight)

        total = sum(probabilities)
        probabilities = [value / total for value in probabilities]
        next_node = rng.choices(neighbors, weights=probabilities, k=1)[0]
        walk.append(next_node)

    return walk


def build_skipgram_pairs(walks, window_size, node_to_id):
    pairs = []
    for walk in walks:
        encoded = [node_to_id[node] for node in walk]
        for index, center in enumerate(encoded):
            left = max(0, index - window_size)
            right = min(len(encoded), index + window_size + 1)
            for context_index in range(left, right):
                if context_index == index:
                    continue
                pairs.append((center, encoded[context_index]))
    return pairs


def train_node2vec_recommender(
    data_dir="ml-latest-small",
    positive_threshold=4.0,
    embedding_dim=32,
    walk_length=12,
    num_walks=8,
    window_size=2,
    num_negative=4,
    epochs=2,
    learning_rate=0.01,
    p=1.0,
    q=0.5,
    seed=42,
):
    movies = load_movies(f"{data_dir}/movies.csv")
    ratings = load_ratings(f"{data_dir}/ratings.csv")
    movie_avg_ratings, movie_rating_counts, _ = compute_movie_rating_stats(ratings)
    global_mean_rating = sum(rating for _, _, rating, _ in ratings) / len(ratings) if ratings else 3.0
    graph = build_interaction_graph(ratings, positive_threshold=positive_threshold)

    nodes = list(graph.nodes())
    node_to_id = {node: index for index, node in enumerate(nodes)}
    id_to_node = {index: node for node, index in node_to_id.items()}
    rng = random.Random(seed)

    walks = []
    for node in nodes:
        for _ in range(num_walks):
            walks.append(generate_node2vec_walk(graph, node, walk_length, p, q, rng))

    pairs = build_skipgram_pairs(walks, window_size, node_to_id)
    if not pairs:
        raise ValueError("Node2Vec training could not generate any training pairs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Node2VecSkipGram(len(nodes), embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Bias sampling toward movie nodes slightly because recommendations are movie-facing.
    sample_weights = []
    for node in nodes:
        sample_weights.append(1.25 if node.startswith("m") else 1.0)
    sample_weights = np.array(sample_weights, dtype=np.float64)
    sample_weights = sample_weights / sample_weights.sum()

    batch_size = 256
    for epoch in range(epochs):
        rng.shuffle(pairs)
        total_loss = 0.0

        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start:batch_start + batch_size]
            centers = torch.tensor([center for center, _ in batch], dtype=torch.long, device=device)
            contexts = torch.tensor([context for _, context in batch], dtype=torch.long, device=device)
            negatives = np.random.choice(
                len(nodes),
                size=(len(batch), num_negative),
                p=sample_weights,
            )
            negatives = torch.tensor(negatives, dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss = model(centers, contexts, negatives)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch)

        average_loss = total_loss / len(pairs)
        print(f"Node2Vec epoch {epoch + 1}/{epochs} complete. Loss: {average_loss:.4f}")

    learned = model.center_embeddings.weight.detach().cpu().numpy()
    movie_embeddings = {}
    for index, vector in enumerate(learned):
        node = id_to_node[index]
        if node.startswith("m"):
            movie_embeddings[int(node[1:])] = vector

    node2vec_model = Node2VecRecommenderModel(
        movies=movies,
        all_items=sorted(movies.keys()),
        embeddings=movie_embeddings,
        movie_avg_ratings=movie_avg_ratings,
        movie_rating_counts=movie_rating_counts,
        global_mean_rating=global_mean_rating,
    )

    print(
        f"Node2Vec model training complete: {len(nodes)} nodes, "
        f"{len(pairs)} training pairs, {len(movie_embeddings)} movie embeddings."
    )
    return node2vec_model


def train_recommender(data_dir="ml-latest-small"):
    return train_heuristic_recommender(data_dir)


def train_all_recommenders(data_dir="ml-latest-small"):
    heuristic = train_heuristic_recommender(data_dir)
    pagerank = train_pagerank_recommender(data_dir)
    node2vec = train_node2vec_recommender(data_dir)
    return RecommenderSuite(
        heuristic=heuristic,
        pagerank=pagerank,
        node2vec=node2vec,
    )


def main():
    data_dir = "ml-latest-small"
    k = 20
    n = 10
    test_ratio = 0.2

    print("=" * 60)
    print("  Movie Recommendation System - Multi-Model Suite")
    print("=" * 60)

    print("\n[1/6] Training heuristic model...")
    heuristic = train_heuristic_recommender(data_dir)

    print("[2/6] Splitting ratings into train / test (80 / 20 by timestamp)...")
    train, test = split_by_timestamp(heuristic.ratings, test_ratio)
    print(f"      Train: {len(train)}  |  Test: {len(test)}")

    print("[3/6] Building collaborative filtering structures...")
    train_matrix = build_user_item_matrix(train)
    train_item_index = build_item_user_index(train_matrix)
    all_items = sorted(heuristic.movies.keys())

    print(f"[4/6] Computing RMSD (k={k})...")
    rmsd = compute_rmsd(test, train_matrix, train_item_index, k=k)
    print(f"      RMSD: {rmsd:.4f}")

    print(f"[5/6] Computing Recall@{n}...")
    recall = compute_recall_at_n(test, train_matrix, train_item_index, all_items, n=n, threshold=4.0)
    print(f"      Recall@{n}: {recall:.4f}")

    print("[6/6] Training PageRank and Node2Vec models...")
    pagerank = train_pagerank_recommender(data_dir)
    node2vec = train_node2vec_recommender(data_dir)

    sample = [1, 296, 593]
    print("\n--- Sample Heuristic Recommendations ---")
    for item in heuristic.recommend_for_selected_movies(sample, n=5):
        print(f"{item['title']} | heuristic score={item['score']:.4f}")

    print("\n--- Sample PageRank Recommendations ---")
    for item in pagerank.recommend_for_selected_movies(sample, n=5):
        print(f"{item['title']} | pagerank score={item['score']:.4f}")

    print("\n--- Sample Node2Vec Recommendations ---")
    for item in node2vec.recommend_for_selected_movies(sample, n=5):
        print(f"{item['title']} | node2vec score={item['score']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
