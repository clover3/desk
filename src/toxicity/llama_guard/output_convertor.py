from typing import List


def convert_predictions_to_binary(predictions: List[str], target_string: str, case_sensitive: bool = False) -> List[
    int]:
    """
    Convert text predictions to binary based on the presence of a target string.

    Args:
    predictions (List[str]): List of text predictions.
    target_string (str): The string to search for in each prediction.
    case_sensitive (bool): Whether the search should be case-sensitive. Defaults to False.

    Returns:
    List[int]: Binary predictions (1 if target_string is found, 0 otherwise).
    """
    binary_predictions = []

    for prediction in predictions:
        if not case_sensitive:
            prediction = prediction.lower()
            target_string = target_string.lower()

        if target_string in prediction:
            binary_predictions.append(1)
        else:
            binary_predictions.append(0)

    return binary_predictions


def parse_predictions(
        text_list: list[str], scores: list[float],
        target_string: str, case_sensitive: bool = False) -> List[tuple[int, float]]:
    paired = zip(text_list, scores)
    return parse_prediction_paired(paired, target_string, case_sensitive)


def parse_prediction_paired(paired, target_string, case_sensitive=False):
    output = []
    for text, score in paired:
        if not case_sensitive:
            text = text.lower()
            target_string = target_string.lower()

        pred = 1 if target_string in text else 0
        score = score if pred else -score
        output.append((pred, score))
    return output
