from typing import Dict


def convert_predictions_to_binary(predictions: Dict[str, str], target_string: str, case_sensitive: bool = False) -> \
        Dict[str, int]:
    binary_predictions = {}
    for id, prediction in predictions.items():
        binary = convert_to_binary(prediction, target_string, case_sensitive)
        binary_predictions[id] = binary
    return binary_predictions


def convert_to_binary(prediction, target_string, case_sensitive=False):
    if not case_sensitive:
        prediction = prediction.lower()
        target_string = target_string.lower()
    binary = 1 if target_string in prediction else 0
    return binary
