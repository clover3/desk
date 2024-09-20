import re
from datetime import datetime
import ast
import sys


def parse_log(file_path):
    results = []
    current_run = None

    with open(file_path, 'r') as file:
        for line in file:
            # Check for new run
            if "__main__ -   __main__" in line:
                if current_run:
                    results.append(current_run)
                current_run = {'timestamp': parse_timestamp(line)}

            # Parse WISEHyperParams
            elif "WISEHyperParams" in line:
                params = parse_hyperparams(line)
                if current_run:
                    current_run['hyperparams'] = params

            # Parse train_acc line
            elif "train_acc" in line:
                metrics = parse_metrics(line)
                if current_run:
                    current_run['metrics'] = metrics

    # Add the last run if it exists
    if current_run:
        results.append(current_run)

    return results


def parse_timestamp(line):
    timestamp_str = line.split(' - ')[0]
    return datetime.strptime(timestamp_str, '%m/%d/%Y %H:%M:%S')


def parse_hyperparams(line):
    # Extract the content inside WISEHyperParams()
    match = re.search(r'WISEHyperParams\((.*)\)', line)
    if not match:
        return {}

    params_str = match.group(1)

    # Replace any single quotes with double quotes for valid JSON
    params_str = params_str.replace("'", '"')

    # Add quotes around keys to make it valid JSON
    params_str = re.sub(r'(\w+)=', r'"\1":', params_str)

    # Wrap the entire string in curly braces to make it a valid dictionary
    params_str = '{' + params_str + '}'

    try:
        # Use ast.literal_eval to safely evaluate the string
        params = ast.literal_eval(params_str)
        return params
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing hyperparameters: {e}")
        return {}


def parse_metrics(line):
    metrics_dict = ast.literal_eval(line.strip())
    return metrics_dict['post']


def parse_value(value):
    try:
        return ast.literal_eval(value)
    except:
        return value


# Example usage
if __name__ == "__main__":
    log_file_path = sys.argv[1]
    parsed_results = parse_log(log_file_path)

    for run in parsed_results:
        weights = run['hyperparams']['inner_params']
        print(weights)
        print(run["metrics"]["train_acc"], run["metrics"]["test_acc"])
        print()