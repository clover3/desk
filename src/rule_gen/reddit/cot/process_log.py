import wandb
import re
import argparse
import os
import sys
from pathlib import Path
import json
import csv
import pandas as pd



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


def upload_to_wandb(metrics_list, project_name, run_name=None, tags=None):
    """
    Upload metrics to Weights & Biases.

    Args:
        metrics_list (list): List of dictionaries containing the metrics
        project_name (str): Name of the W&B project
        run_name (str, optional): Name of the W&B run
        tags (list, optional): List of tags for the W&B run
    """
    # Initialize a W&B run
    run = wandb.init(project=project_name, name=run_name, tags=tags, reinit=True)

    # Check if metrics have step information
    has_step = any('step' in metrics for metrics in metrics_list)

    # If no step information, add it
    if not has_step:
        for i, metrics in enumerate(metrics_list):
            metrics['step'] = i

    # Log each set of metrics
    for metrics in metrics_list:
        step = metrics.get('step', 0)

        # Create a copy of metrics without step for logging
        log_metrics = {k: v for k, v in metrics.items() if k != 'step'}

        # Log to W&B
        wandb.log(log_metrics, step=step)

    # Finish the run
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Parse log files and upload metrics to Weights & Biases')

    parser.add_argument('--file_path', type=str, help='Path to the log file')
    parser.add_argument('--project', type=str, required=True,
                        help='Name of the W&B project')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name of the W&B run')
    parser.add_argument('--tags', type=str, nargs='+', default=None,
                        help='Tags for the W&B run')

    print(sys.argv)
    args = parser.parse_args(sys.argv[1:])

    # Check if W&B API key is set
    # if 'WANDB_API_KEY' not in os.environ:
    #     api_key = input("Enter your Weights & Biases API key: ")
    #     os.environ['WANDB_API_KEY'] = api_key

    # Parse the log file
    metrics_list = parse_training_log(args.file_path)

    if not metrics_list:
        print(f"Error: No metrics found in {args.file_path}")
        return

    print(f"Found {len(metrics_list)} sets of metrics")

    # Upload to W&B
    upload_to_wandb(metrics_list, args.project, args.run_name, args.tags)
    #
    print(f"Successfully uploaded metrics to W&B project '{args.project}'")


if __name__ == '__main__':
    main()