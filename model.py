"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np

def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""

    if len(labels) == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)

    probabilities = counts / len(labels)

    gini = 1 - np.sum(probabilities ** 2)

    return float(gini)

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    mask = features[:, feature_index] <= threshold

    left_features = features[mask]
    left_labels = labels[mask]

    right_features = features[~mask]
    right_labels = labels[~mask]

    return left_features,left_labels,right_features,right_labels

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    parent_impurity = impurity(parent_labels)
    left_impurity = impurity(left_labels)
    right_impurity = impurity(right_labels)

    n = len(parent_labels)
    left_weight = len(left_labels) / n
    right_weight = len(right_labels) / n

    children_impurity = (
        left_weight * left_impurity +
        right_weight * right_impurity
    )

    return float(parent_impurity - children_impurity)

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    best_feature = None
    best_threshold = None
    best_score = 0.0

    for feature_index in feature_indices:
        values = np.unique(features[:, feature_index])

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            left_features, left_labels, right_features, right_labels = split_dataset(
                features, labels, feature_index, threshold
            )

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            score = split_score(labels, left_labels, right_labels)

            if score > best_score:
                best_score = score
                best_feature = feature_index
                best_threshold = threshold

    return {
        "feature_index": best_feature,
        "threshold": best_threshold,
        "score": float(best_score)
    }

# Step 5 - should_stop
import numpy as np

def should_stop(labels, depth, max_depth, min_samples_split):
    if len(np.unique(labels)) == 1:
        return True

    if depth >= max_depth:
        return True

    if len(labels) < min_samples_split:
        return True

    return False

# Step 6 - leaf_prediction
import numpy as np

def leaf_prediction(labels):
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])

# Step 7 - build_tree
import numpy as np

def build_tree(features, labels, depth=0, max_depth=10, min_samples_split=2, feature_subset=None):
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {
            'leaf': True,
            'prediction': leaf_prediction(labels)
        }

    if feature_subset is None:
        feature_indices = range(features.shape[1])
    else:
        feature_indices = feature_subset

    split = best_split(features, labels, feature_indices)

    if split['feature_index'] is None or split['score'] <= 0:
        return {
            'leaf': True,
            'prediction': leaf_prediction(labels)
        }

    feature_index = split['feature_index']
    threshold = split['threshold']

    left_features, left_labels, right_features, right_labels = split_dataset(
        features, labels, feature_index, threshold
    )

    if len(left_labels) == 0 or len(right_labels) == 0:
        return {
            'leaf': True,
            'prediction': leaf_prediction(labels)
        }

    return {
        'leaf': False,
        'feature_index': int(feature_index),
        'threshold': float(threshold),
        'left': build_tree(
            left_features, left_labels, depth + 1,
            max_depth, min_samples_split, feature_subset
        ),
        'right': build_tree(
            right_features, right_labels, depth + 1,
            max_depth, min_samples_split, feature_subset
        )
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    while not tree['leaf']:
        feature_index = tree['feature_index']
        threshold = tree['threshold']

        if example[feature_index] <= threshold:
            tree = tree['left']
        else:
            tree = tree['right']

    return int(tree['prediction'])

# Step 9 - predict_tree
import numpy as np

def predict_tree(tree, features):
    return np.array([
        predict_example_tree(tree, example)
        for example in features
    ], dtype=int)

# Step 10 - bootstrap_sample
import numpy as np

def bootstrap_sample(features, labels, rng):
    n = len(features)
    indices = rng.integers(0, n, size=n)

    sampled_features = features[indices]
    sampled_labels = labels[indices]

    return sampled_features, sampled_labels

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    return rng.choice(num_features, size=num_to_pick, replace=False)

# Step 12 - train_forest
import numpy as np

def train_forest(
    X,
    y,
    num_trees,
    num_features_per_split=None,
    max_depth=10,
    min_samples_split=2,
    random_state=None
):
    n_samples, n_features = X.shape

    if num_features_per_split is None:
        num_features_per_split = max(1, round(np.sqrt(n_features)))

    rng = np.random.default_rng(random_state)
    forest = []

    for _ in range(num_trees):
        X_bootstrap, y_bootstrap = bootstrap_sample(X, y, rng)

        feature_indices = feature_subset(
            n_features, num_features_per_split, rng
        )

        tree = build_tree(
            X_bootstrap,
            y_bootstrap,
            depth=0,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            feature_subset=feature_indices
        )

        forest.append({
            "tree": tree,
            "feature_indices": feature_indices
        })

    return forest

# Step 13 - combine_predictions
import numpy as np

def combine_predictions(tree_predictions):
    n_trees, n_samples = tree_predictions.shape
    predictions = np.empty(n_samples, dtype=int)

    for i in range(n_samples):
        classes, counts = np.unique(
            tree_predictions[:, i],
            return_counts=True
        )

        predictions[i] = classes[np.argmax(counts)]

    return predictions

# Step 14 - predict_forest
import numpy as np

def predict_forest(forest, features):
    if len(forest) == 0:
        return np.array([], dtype=int)

    all_predictions = []

    for item in forest:
        tree = item["tree"]

        predictions = predict_tree(tree, features)
        all_predictions.append(predictions)

    all_predictions = np.array(all_predictions)

    return np.asarray(combine_predictions(all_predictions), dtype=int)

# Step 15 - accuracy (not yet solved)
# TODO: implement

