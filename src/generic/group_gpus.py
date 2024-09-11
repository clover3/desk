from generic.gpu_count import run_command


def parse_dash_indicated_columns(output):
    lines = output.split('\n')
    header = lines[0]
    dashes = lines[1]

    # Determine column positions based on the dashed line
    column_positions = [0]
    print(dashes)
    for i, char in enumerate(dashes):
        if char == ' ' and i + 1 < len(dashes) and dashes[i + 1] == '-':
            column_positions.append(i + 1)
    column_positions.append(len(dashes))
    print(column_positions)

    # Function to split a line based on column positions
    def split_line(line):
        return [line[start:end].strip() for start, end in zip(column_positions[:-1], column_positions[1:])]

    # Parse header
    header_fields = split_line(header)
    print(header_fields)

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
    print(users)



if __name__ == "__main__":
    main()