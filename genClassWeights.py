"""
Generate class weights (class_weights) for visualization codes.
"""

# ----- Define categories and codes -----
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

# ----- Assign default weights to each category -----
default_weights = {
    "roads":     1.2,
    "water":     1.3,
    "buildings": 2.0,
    "terrain":   0.9
}

# ----- Optional overrides for specific codes -----
overrides = {
    # code: weight
    # 10710: 2.5,
    # 12400: 3.0
}

# ----- Group them for easier processing -----
categories = {
    "roads":     roads,
    "water":     water,
    "buildings": buildings,
    "terrain":   terrain
}


def main():
    """Write class weights for each code into a SQL file."""
    output_file = "class_weights.sql"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("BEGIN;\n\n")
            f.write(
                "CREATE TABLE IF NOT EXISTS class_weights (\n"
                "    code   INT PRIMARY KEY,\n"
                "    weight DOUBLE PRECISION\n"
                ");\n\n"
            )

            for category_name, codes in categories.items():
                cat_weight = default_weights.get(category_name, 1.0)
                for code in codes:
                    # If there's an override for this code, use that
                    weight_val = overrides.get(code, cat_weight)
                    f.write(
                        "INSERT INTO class_weights (code, weight) "
                        f"VALUES ({code}, {weight_val}) "
                        "ON CONFLICT (code) DO UPDATE SET weight = EXCLUDED.weight;\n"
                    )

            f.write("\nCOMMIT;\n")

        print(f"Generated {output_file} successfully.")

    except IOError as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":
    main()
