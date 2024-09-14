from generic.gpu_count import run_command, parse_squeue


def parse_dash_indicated_columns(output):
    lines = output.split('\n')
    header = lines[0]
    dashes = lines[1]

    # Determine column positions based on the dashed line
    column_positions = [0]
    for i, char in enumerate(dashes):
        if char == ' ' and i + 1 < len(dashes) and dashes[i + 1] == '-':
            column_positions.append(i + 1)
    column_positions.append(len(dashes))

    # Function to split a line based on column positions
    def split_line(line):
        return [line[start:end].strip() for start, end in zip(column_positions[:-1], column_positions[1:])]

    # Parse header
    header_fields = split_line(header)

    # Parse data rows
    parsed_data = []
    for line in lines[2:]:  # Skip header and dashes
        values = split_line(line)
        row_dict = dict(zip(header_fields, values))
        parsed_data.append(row_dict)

    return parsed_data


def main():
    output = run_command('sacctmgr show associations account=hartvigsen_lab')
    rows = parse_dash_indicated_columns(output)
    users = [r['User'] for r in rows if r['User']]
    print("users", users)

    no_gpu_users = []
    for user in users:
        squeue_output = run_command(f"squeue -u {user} -t R -o \"%.18i %.9P %.8j %.8u %.10M %.6D %R %b\"")
        gpu_usage = parse_squeue(squeue_output)
        n_gpu = sum(gpu_usage.values())
        if n_gpu:
            print(f"{user}\t{n_gpu}")
        else:
            no_gpu_users.append(user)

    print("no_gpu_users", no_gpu_users)


if __name__ == "__main__":
    main()