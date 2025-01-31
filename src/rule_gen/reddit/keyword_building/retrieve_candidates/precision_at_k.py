from sklearn.metrics import precision_score


def precision_at_ks(y_true, sorted_indices, ks):
    y_true_sorted = [y_true[i] for i in sorted_indices]

    # Initialize a dictionary to store precision-at-k results
    precision_results = {}

    # Iterate over each k value
    for k in ks:
        # Consider only the top-k results
        y_true_top_k = y_true_sorted[:k]

        # Calculate precision for the top-k results
        precision = precision_score(y_true_top_k, [1] * len(y_true_top_k), zero_division=0)

        # Store the result in the dictionary
        precision_results[k] = precision

    return precision_results
