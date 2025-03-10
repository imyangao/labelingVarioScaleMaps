"""
Generate a compatibility matrix (class_comp_matrix) between visualization codes.
"""

# -----  Define categories and codes -----
roads = [
    # 10000–10830
    10000, 10100, 10200, 10300, 10310, 10400, 10410, 10500, 10510,
    10600, 10700, 10710, 10720, 10730, 10740, 10750, 10760, 10770,
    10780, 10790, 10800, 10820, 10830
]

water = [
    # 12000–12700
    12000, 12100, 12200, 12300, 12400, 12410, 12420, 12430,
    12500, 12600, 12610, 12700
]

buildings = [
    # 13000–13400
    13000, 13100, 13200, 13300, 13400
]

terrain = [
    # 14000–14210
    14000, 14010, 14020, 14030, 14040, 14050, 14060, 14070, 14080, 14090,
    14100, 14110, 14120, 14130, 14140, 14160, 14170, 14180, 14190,
    14200, 14210
]

all_codes = roads + water + buildings + terrain

# ----- Define groups for lookups -----
groups = {
    'roads': set(roads),
    'water': set(water),
    'buildings': set(buildings),
    'terrain': set(terrain),
}

# ----- Define scoring rules -----
SAME_CODE_SCORE = 1.0
SAME_GROUP_SCORE = 0.8
DIFF_GROUP_SCORE = 0.2

# Optional overrides for special combinations
compat_exceptions = {
    (10000, 12500): 0.05,  # example: runway + large water
    (10300, 10310): 0.9,   # example: hoofdweg variants
    (14060, 14090): 0.9,   # example: different forest types
    # ...
}


def find_group(code):
    """Return the name of the group to which a code belongs, or None if not found."""
    for group_name, code_set in groups.items():
        if code in code_set:
            return group_name
    return None


def main():
    """Write all code compatibilities into a SQL file."""
    output_file = "class_comp_matrix.sql"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("BEGIN;\n")
            f.write("CREATE TABLE IF NOT EXISTS class_comp_matrix (code_from INT, code_to INT, comp_value FLOAT);\n")

            for code_from in all_codes:
                group_from = find_group(code_from)
                for code_to in all_codes:
                    group_to = find_group(code_to)

                    # Base score
                    if code_from == code_to:
                        score = SAME_CODE_SCORE
                    elif group_from == group_to:
                        score = SAME_GROUP_SCORE
                    else:
                        score = DIFF_GROUP_SCORE

                    # Check special exceptions
                    if (code_from, code_to) in compat_exceptions:
                        score = compat_exceptions[(code_from, code_to)]
                    elif (code_to, code_from) in compat_exceptions:
                        score = compat_exceptions[(code_to, code_from)]

                    f.write(
                        f"INSERT INTO class_comp_matrix (code_from, code_to, comp_value)\n"
                        f"    VALUES ({code_from}, {code_to}, {score});\n"
                    )

            f.write("COMMIT;\n")

        print(f"Generated {output_file} successfully.")

    except IOError as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":
    main()