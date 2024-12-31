import sys

from desk_util.open_ai import get_open_ai


def main():
    batch_id = sys.argv[1]
    client = get_open_ai()
    batch = client.batches.retrieve(batch_id)
    print(batch)


if __name__ == "__main__":
    main()