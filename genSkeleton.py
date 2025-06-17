from labeling_core.skeleton import generate_skeleton_for_gpkg

if __name__ == "__main__":
    # This script remains as an example or for direct execution,
    # but all core logic is now in the labeling_core package.
    input_gpkg = r"C:\topnl_test\yan_topo2geom_2500_enriched.gpkg"
    output_gpkg = r"skeleton_2500_test.gpkg"

    print("Running skeleton generation...")
    generate_skeleton_for_gpkg(
        input_gpkg,
        output_gpkg,
        do_simplify=False,  # enable polygon simplification
        simplify_tolerance=0.0  # set a tolerance
    )
    print("Skeleton generation complete.")