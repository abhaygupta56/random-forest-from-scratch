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

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

