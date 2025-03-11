def remove_substring_duplicates(strings):
    strings = sorted(strings, key=len)  # Sort by length to prioritize shorter strings
    unique_strings = []

    for s in strings:
        if not any(s in other for other in unique_strings):
            unique_strings.append(s)

    return unique_strings


