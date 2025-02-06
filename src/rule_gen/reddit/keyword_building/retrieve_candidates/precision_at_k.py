from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



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


def recall_at_ks(y_true, sorted_indices, ks):
    y_true_sorted = [y_true[i] for i in sorted_indices]
    total_relevant = sum(y_true)  # Total number of relevant items

    # Initialize a dictionary to store recall-at-k results
    recall_results = {}

    # Iterate over each k value
    for k in ks:
        # Consider only the top-k results
        y_true_top_k = y_true_sorted[:k]

        # Calculate number of relevant items in top-k
        relevant_in_top_k = sum(y_true_top_k)

        # Calculate recall: relevant items in top-k / total relevant items
        recall = relevant_in_top_k / total_relevant if total_relevant > 0 else 0

        # Store the result in the dictionary
        recall_results[k] = recall

    return recall_results