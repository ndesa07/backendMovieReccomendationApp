# =============================================================================
# Movie Recommendation System - User-Based Collaborative Filtering
# 42913 Social and Information Network Analysis - Group Project
# Dataset: MovieLens 100k (ml-100k/u.data)
# No third-party imports - pure Python standard library only
# =============================================================================

# --------------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------------

def load_ratings(filepath):
    """
    Load ratings from the u.data file.
    Format: user_id TAB item_id TAB rating TAB timestamp
    Returns a list of (user_id, item_id, rating, timestamp) tuples.
    """
    ratings = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 4:
                user_id  = int(parts[0])
                item_id  = int(parts[1])
                rating   = float(parts[2])
                timestamp = int(parts[3])
                ratings.append((user_id, item_id, rating, timestamp))
    return ratings


def load_movies(filepath):
    """
    Load movie metadata from u.item.
    Returns a dict: {movie_id: movie_title}
    """
    movies = {}
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 2:
                movie_id    = int(parts[0])
                movie_title = parts[1]
                movies[movie_id] = movie_title
    return movies


# --------------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# --------------------------------------------------------------------------

def split_by_timestamp(ratings, test_ratio=0.2):
    """
    Split ratings chronologically: the most recent (test_ratio * 100)%
    of interactions form the test set.
    """
    sorted_ratings = sorted(ratings, key=lambda x: x[3])
    split_idx = int(len(sorted_ratings) * (1.0 - test_ratio))
    train = sorted_ratings[:split_idx]
    test  = sorted_ratings[split_idx:]
    return train, test


# --------------------------------------------------------------------------
# 3. DATA STRUCTURES
# --------------------------------------------------------------------------

def build_user_item_matrix(ratings):
    """
    Build a nested dict: {user_id: {item_id: rating}}.
    """
    matrix = {}
    for user_id, item_id, rating, _ in ratings:
        if user_id not in matrix:
            matrix[user_id] = {}
        matrix[user_id][item_id] = rating
    return matrix


def build_item_user_index(matrix):
    """
    Inverted index: {item_id: set_of_user_ids} for fast neighbour lookup.
    """
    index = {}
    for user_id, items in matrix.items():
        for item_id in items:
            if item_id not in index:
                index[item_id] = set()
            index[item_id].add(user_id)
    return index


# --------------------------------------------------------------------------
# 4. SIMILARITY
# --------------------------------------------------------------------------

def user_mean(user_ratings):
    """Return the mean rating for a user (dict of {item_id: rating})."""
    if not user_ratings:
        return 0.0
    total = 0.0
    for r in user_ratings.values():
        total += r
    return total / len(user_ratings)


def pearson_correlation(ratings_a, ratings_b):
    """
    Pearson correlation between two users computed over their co-rated items.
    Returns a float in [-1, 1]; returns 0.0 if fewer than 2 co-rated items.
    """
    # Find co-rated items
    common = [i for i in ratings_a if i in ratings_b]
    n = len(common)
    if n < 2:
        return 0.0

    mean_a = sum(ratings_a[i] for i in common) / n
    mean_b = sum(ratings_b[i] for i in common) / n

    numerator   = 0.0
    sq_dev_a    = 0.0
    sq_dev_b    = 0.0
    for i in common:
        da = ratings_a[i] - mean_a
        db = ratings_b[i] - mean_b
        numerator  += da * db
        sq_dev_a   += da * da
        sq_dev_b   += db * db

    denominator = (sq_dev_a ** 0.5) * (sq_dev_b ** 0.5)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


# --------------------------------------------------------------------------
# 5. RATING PREDICTION
# --------------------------------------------------------------------------

def predict_rating(user_id, item_id, matrix, item_user_index,
                   k=20, min_support=2):
    """
    Predict the rating user_id would give to item_id using user-based CF.

    Algorithm:
      1. Find candidate neighbours - users who also rated item_id.
      2. Compute Pearson similarity between the target user and each candidate.
      3. Select the top-k most similar neighbours (positive similarity only).
      4. Compute a weighted deviation prediction:
            pred = mean(u) + sum(sim * (r_v - mean(v))) / sum(|sim|)

    Falls back to the target user's mean (or global mean 3.0) when no
    neighbours are available.
    """
    # Fallback values
    if user_id not in matrix:
        return 3.0
    user_ratings = matrix[user_id]
    u_mean       = user_mean(user_ratings)

    # Candidate neighbours: users who rated item_id
    candidates = item_user_index.get(item_id, set())

    neighbours = []
    for other_id in candidates:
        if other_id == user_id:
            continue
        other_ratings = matrix[other_id]
        sim = pearson_correlation(user_ratings, other_ratings)
        if sim > 0:
            neighbours.append((sim, other_id))

    if not neighbours:
        return u_mean

    # Sort descending by similarity, keep top-k
    neighbours.sort(reverse=True)
    top_k = neighbours[:k]

    numerator   = 0.0
    denominator = 0.0
    for sim, other_id in top_k:
        other_ratings = matrix[other_id]
        v_mean = user_mean(other_ratings)
        numerator   += sim * (other_ratings[item_id] - v_mean)
        denominator += abs(sim)

    if denominator == 0.0:
        return u_mean

    prediction = u_mean + numerator / denominator
    # Clamp to valid rating range [1, 5]
    return max(1.0, min(5.0, prediction))


# --------------------------------------------------------------------------
# 6. EVALUATION METRICS
# --------------------------------------------------------------------------

def compute_rmsd(test_data, matrix, item_user_index, k=20):
    """
    Root-Mean-Square Deviation between predicted and actual ratings
    over the test set.
    """
    squared_errors = []
    for user_id, item_id, actual, _ in test_data:
        predicted = predict_rating(user_id, item_id, matrix,
                                   item_user_index, k)
        squared_errors.append((predicted - actual) ** 2)

    if not squared_errors:
        return 0.0
    mean_sq_err = sum(squared_errors) / len(squared_errors)
    return mean_sq_err ** 0.5


def recommend_top_n(user_id, matrix, item_user_index, all_items, n=10):
    """
    Return the top-n item IDs predicted for user_id,
    excluding items already rated.
    """
    if user_id not in matrix:
        return []
    rated = set(matrix[user_id].keys())

    predictions = []
    for item_id in all_items:
        if item_id in rated:
            continue
        pred = predict_rating(user_id, item_id, matrix,
                              item_user_index, k=20)
        predictions.append((pred, item_id))

    predictions.sort(reverse=True)
    return [item_id for _, item_id in predictions[:n]]


def compute_recall_at_n(test_data, matrix, item_user_index,
                         all_items, n=10, threshold=4.0):
    """
    Average Recall@N across users.

    Relevant items = test items rated >= threshold.
    Recall for one user = |recommended ∩ relevant| / |relevant|
    """
    # Build per-user relevant items from test set
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
        recommended = set(recommend_top_n(user_id, matrix,
                                          item_user_index, all_items, n))
        tp = len(recommended & relevant)
        recalls.append(tp / len(relevant))

    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


# --------------------------------------------------------------------------
# 7. MAIN ENTRY POINT
# --------------------------------------------------------------------------

def main():
    DATA_FILE   = 'ml-100k/u.data'
    MOVIES_FILE = 'ml-100k/u.item'
    K           = 20      # number of neighbours
    N           = 10      # recommendation list length
    TEST_RATIO  = 0.2

    print("=" * 60)
    print("  Movie Recommendation System - User-Based CF")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/5] Loading data...")
    ratings = load_ratings(DATA_FILE)
    movies  = load_movies(MOVIES_FILE)
    print(f"      {len(ratings)} ratings | "
          f"{len(set(u for u,_,_,_ in ratings))} users | "
          f"{len(set(i for _,i,_,_ in ratings))} movies")

    # --- Split ---
    print("[2/5] Splitting into train / test (80 / 20 by timestamp)...")
    train, test = split_by_timestamp(ratings, TEST_RATIO)
    print(f"      Train: {len(train)}  |  Test: {len(test)}")

    # --- Build matrix ---
    print("[3/5] Building user-item matrix and item index...")
    matrix          = build_user_item_matrix(train)
    item_user_index = build_item_user_index(matrix)
    all_items       = list(set(i for _, i, _, _ in train))

    # --- RMSD ---
    print(f"[4/5] Computing RMSD (k={K}, evaluating {len(test)} test samples)...")
    rmsd = compute_rmsd(test, matrix, item_user_index, k=K)
    print(f"\n  >> RMSD (k={K}): {rmsd:.4f}")

    # --- Recall ---
    print(f"[5/5] Computing Recall@{N} (rating threshold >= 4.0)...")
    recall = compute_recall_at_n(test, matrix, item_user_index,
                                  all_items, n=N, threshold=4.0)
    print(f"  >> Recall@{N}: {recall:.4f}")

    # --- Effect of k ---
    print("\n--- Effect of k on RMSD ---")
    print(f"{'k':>6}  {'RMSD':>8}")
    for k_val in [5, 10, 20, 30, 50]:
        r = compute_rmsd(test, matrix, item_user_index, k=k_val)
        print(f"{k_val:>6}  {r:>8.4f}")

    # --- Sample recommendations ---
    print("\n--- Sample Recommendations for User 1 ---")
    top10 = recommend_top_n(1, matrix, item_user_index, all_items, n=10)
    print(f"{'Rank':>5}  {'Movie ID':>9}  Title")
    print("-" * 50)
    for rank, movie_id in enumerate(top10, start=1):
        title = movies.get(movie_id, "Unknown")
        print(f"{rank:>5}  {movie_id:>9}  {title}")

    print("\nDone.")


if __name__ == "__main__":
    main()
