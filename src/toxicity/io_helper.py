import csv
from typing import List, Tuple
from chair.misc_lib import make_parent_exists


import json


def save_csv(tuple_itr, file_path: str) -> None:
    make_parent_exists(file_path)
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in tuple_itr:
            writer.writerow(row)


def save_text_list_as_csv(text_itr, file_path: str) -> None:
    make_parent_exists(file_path)
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for text in text_itr:
            writer.writerow([text])
            f.flush()


def read_csv(file_path: str):
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        return list(csv.reader(f))


def read_csv_column(file_path: str, column_i):
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        output = []
        for row in csv.reader(f):
            output.append(row[column_i])
        return output


def load_two_column_csv(file_path: str) -> Tuple[List[str], List[str]]:
    ids = []
    predictions = []

    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) == 2:
                id, prediction = row
                ids.append(id)
                predictions.append(prediction)
            else:
                print(f"Skipping malformed row: {row}")

    print(f"Loaded {len(ids)} items from {file_path}")
    return ids, predictions


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
