import re
import argparse
from pathlib import Path
import json


def parse_training_log(file_path):
    """
    Parse a training log file containing metrics output line by line.

    Args:
        file_path (str): Path to the log file

    Returns:
        list: List of dictionaries containing parsed metrics for each step
    """
    results = []
    current_step = None
    current_data = {}

    # Process file line by line
    with open(file_path, 'r') as f:
        for line in f:
            # Check if this line contains a step identifier
            step_match = re.search(r'step:(\d+)', line)
            if step_match:
                # If we have data from a previous step, save it
                if current_step is not None and current_data:
                    current_data['step'] = current_step
                    results.append(current_data)

                # Start a new step
                current_step = int(step_match.group(1))
                current_data = {}

            # Extract metrics from the current line
            metric_matches = re.finditer(r'(\w+(?:/\w+)*):(-?\d+\.\d+)', line)
            for match in metric_matches:
                metric_name = match.group(1)
                metric_value = float(match.group(2))
                current_data[metric_name] = metric_value

    # Add the last step if it exists
    if current_step is not None and current_data:
        current_data['step'] = current_step
        results.append(current_data)

    return results


def main():
    parser = argparse.ArgumentParser(description='Parse training metrics log file')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path', default=None)

    args = parser.parse_args()

    # Parse the log file
    results = parse_training_log(args.log_file)

    # Display summary
    print(f"Parsed {len(results)} steps.")

    if results:
        # Get metrics from the first entry to show what's available
        metrics = list(results[0].keys())
        print(f"\nMetrics found: {', '.join(sorted(metrics))}")

    # Save to JSON if output specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    else:
        # Print the first result
        if results:
            print("\nFirst step data:")
            print(json.dumps(results[0], indent=2))
            print("\nLast step data:")
            print(json.dumps(results[-1], indent=2))
            print(f"\nTotal steps: {len(results)}")


if __name__ == "__main__":
    main()