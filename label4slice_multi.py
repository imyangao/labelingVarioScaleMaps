import psycopg2
import os
import fiona
import geopandas as gpd
import math
from shapely.geometry import Point, LineString
from shapely.algorithms.polylabel import polylabel
from sqlalchemy import create_engine
from urllib.parse import quote_plus

from scalestep import ScaleStep
from genSkeleton import (
    build_skeleton_lines, 
    lines_to_graph, 
    get_junction_nodes, 
    find_junction_to_junction_paths, 
    merge_collinear_lines,
    largest_inscribed_circle,
    generate_skeleton_for_gpkg
)


################################################################################
# Database & table setup
################################################################################
DB_NAME = "tgap_test"
DB_USER = "postgres"
DB_PASS = "Gy@001130"
DB_HOST = "localhost"
DB_PORT = 5432


# Use quote_plus to properly URL-encode the password
escaped_password = quote_plus(DB_PASS)
DB_URI = f"postgresql://{DB_USER}:{escaped_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URI)


def create_or_reset_anchors_table(conn):
    """
    Create (or reset) a single table named label_anchors_from_slices,
    in which we store anchors from multiple steps (slices).
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS label_anchors_from_slices;")
        cur.execute("""
            CREATE TABLE label_anchors_from_slices (
                label_id   SERIAL PRIMARY KEY,
                step_value INTEGER,
                face_id    INTEGER,
                feature_class INTEGER,
                name TEXT,
                anchor_geom geometry(POINT, 28992),
                angle      DOUBLE PRECISION
            );
        """)
    conn.commit()


################################################################################
# Process geometries directly without intermediate files
################################################################################
def process_geometries_directly(conn, step_value, do_simplify=True, simplify_tolerance=10.0):
    """
    Process geometries directly from the database for a given step_value,
    compute anchor points based on feature_class, and insert them into the anchors table.
    """
    # Query to get all faces at the specified step
    sql = f"""
    WITH polygonized_edges AS (
        SELECT
            (ST_Dump(ST_Polygonize(e.geometry))).geom::geometry(Polygon, 28992) AS polygon_geom
        FROM yan_tgap_edge e
        WHERE e.step_low <= {step_value} AND e.step_high > {step_value}
    )
    SELECT
        p.polygon_geom,
        f.face_id,
        f.feature_class,
        ff.name
    FROM polygonized_edges p
    JOIN yan_tgap_face f
      ON ST_Contains(p.polygon_geom, f.pip_geometry)
    LEFT JOIN yan_face ff
      ON ff.face_id = f.face_id
    WHERE f.step_low <= {step_value} 
      AND f.step_high > {step_value};
    """
    
    # Read the data into a GeoDataFrame
    gdf = gpd.read_postgis(sql, engine, geom_col="polygon_geom")
    
    if gdf.empty:
        print(f"No data found for step_value {step_value}")
        return
    
    # Process each geometry based on its feature_class
    for idx, row in gdf.iterrows():
        try:
            # Access the geometry directly from the row
            polygon = row['polygon_geom'].buffer(0)  # Fix geometry
            feature_class = row.get('feature_class', None)
            face_id = row.get('face_id', None)
            name = row.get('name', None)
            
            # Skip if feature_class is missing or invalid
            if feature_class is None:
                continue
                
            # Process based on feature_class
            if 10000 <= feature_class < 11000:  # Roads
                process_road_geometry(conn, step_value, polygon, feature_class, face_id, name, 
                                     do_simplify, simplify_tolerance)
            elif 12000 <= feature_class < 13000:  # Water
                process_water_geometry(conn, step_value, polygon, feature_class, face_id, name, 
                                       do_simplify, simplify_tolerance)
            elif 13000 <= feature_class < 14000:  # Buildings
                process_building_geometry(conn, step_value, polygon, feature_class, face_id, name)
                
        except Exception as e:
            print(f"Error processing geometry at index {idx}: {e}")


def process_road_geometry(conn, step_value, polygon, feature_class, face_id, name, 
                         do_simplify, simplify_tolerance):
    """Process road geometry and insert anchor points into the database."""
    # Optional polygon simplification
    if do_simplify and simplify_tolerance > 0:
        poly_for_skel = polygon.simplify(simplify_tolerance, preserve_topology=True)
    else:
        poly_for_skel = polygon

    # Build raw skeleton
    raw_skel_lines = build_skeleton_lines(poly_for_skel)
    if not raw_skel_lines:
        return

    # Convert to graph -> find junctions -> connect them
    G = lines_to_graph(raw_skel_lines)
    junctions = get_junction_nodes(G, min_degree=3)
    if len(junctions) < 2:
        # not enough skeleton complexity
        return

    primary_paths = find_junction_to_junction_paths(G, junctions)
    merged_primary = merge_collinear_lines(primary_paths, angle_threshold=5.0)

    # LABELING: Take all lines above a fraction of the max length
    sorted_lines = sorted(merged_primary, key=lambda l: l.length, reverse=True)
    if len(sorted_lines) > 0:
        max_length = sorted_lines[0].length
        threshold_fraction = 0.25  # label lines >= 25% of the longest line
        length_threshold = threshold_fraction * max_length
    else:
        length_threshold = 0.0

    # Insert anchor points into the database
    insert_sql = """
    INSERT INTO label_anchors_from_slices
    (step_value, face_id, feature_class, name, anchor_geom, angle)
    VALUES (%s, %s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 28992), %s);
    """
    
    with conn.cursor() as cur:
        for line in sorted_lines:
            if line.length < length_threshold:
                break  # lines are sorted desc, so break as soon as below threshold

            # Compute midpoint and angle
            midpoint = line.interpolate(0.5, normalized=True)
            (x1, y1) = line.coords[0]
            (x2, y2) = line.coords[-1]
            dx = x2 - x1
            dy = y2 - y1
            angle_deg = math.degrees(math.atan2(dy, dx))
            # Constrain angle to -90..90
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180

            # Insert into database
            cur.execute(insert_sql, (
                step_value,
                face_id,
                feature_class,
                name,
                midpoint.wkt,
                angle_deg
            ))
    
    conn.commit()


def process_water_geometry(conn, step_value, polygon, feature_class, face_id, name, 
                          do_simplify, simplify_tolerance):
    """Process water geometry and insert anchor points into the database."""
    # Optional polygon simplification
    if do_simplify and simplify_tolerance > 0:
        poly_for_skel = polygon.simplify(simplify_tolerance, preserve_topology=True)
    else:
        poly_for_skel = polygon

    # Build raw skeleton
    raw_skel_lines = build_skeleton_lines(poly_for_skel)
    if not raw_skel_lines:
        return

    # Convert to graph -> find junctions -> connect them
    G = lines_to_graph(raw_skel_lines)
    junctions = get_junction_nodes(G, min_degree=3)
    if len(junctions) < 2:
        return

    primary_paths = find_junction_to_junction_paths(G, junctions)
    merged_primary = merge_collinear_lines(primary_paths, angle_threshold=5.0)

    # LABELING: multiple lines above length threshold
    sorted_lines = sorted(merged_primary, key=lambda l: l.length, reverse=True)
    if len(sorted_lines) > 0:
        max_length = sorted_lines[0].length
        threshold_fraction = 0.25
        length_threshold = threshold_fraction * max_length
    else:
        length_threshold = 0.0

    # Insert anchor points into the database
    insert_sql = """
    INSERT INTO label_anchors_from_slices
    (step_value, face_id, feature_class, name, anchor_geom, angle)
    VALUES (%s, %s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 28992), %s);
    """
    
    with conn.cursor() as cur:
        for line in sorted_lines:
            if line.length < length_threshold:
                break

            midpoint = line.interpolate(0.5, normalized=True)
            (x1, y1) = line.coords[0]
            (x2, y2) = line.coords[-1]
            dx = x2 - x1
            dy = y2 - y1
            angle_deg = math.degrees(math.atan2(dy, dx))
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180

            # Insert into database
            cur.execute(insert_sql, (
                step_value,
                face_id,
                feature_class,
                name,
                midpoint.wkt,
                angle_deg
            ))
    
    conn.commit()


def process_building_geometry(conn, step_value, polygon, feature_class, face_id, name):
    """Process building geometry and insert anchor points into the database."""
    center, radius = largest_inscribed_circle(polygon)
    if center is not None:
        # Calculate rotation angle from minimum rotated rectangle
        rect = polygon.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
        max_length = 0.0
        best_angle = 0.0
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            length = Point(p1).distance(Point(p2))
            if length > max_length:
                max_length = length
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                angle_rad = math.atan2(dy, dx)
                best_angle = math.degrees(angle_rad)
        # Adjust angle to be within -90 to 90 degrees
        if best_angle > 90:
            best_angle -= 180
        elif best_angle < -90:
            best_angle += 180

        # Insert into database
        insert_sql = """
        INSERT INTO label_anchors_from_slices
        (step_value, face_id, feature_class, name, anchor_geom, angle)
        VALUES (%s, %s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 28992), %s);
        """
        
        with conn.cursor() as cur:
            cur.execute(insert_sql, (
                step_value,
                face_id,
                feature_class,
                name,
                center.wkt,
                best_angle
            ))
        
        conn.commit()


################################################################################
# Optional: Create intermediate files if needed
################################################################################
def create_slice_table(conn, step_value):
    """
    Creates a slice table for the specified step_value
    """
    table_name = f"yan_topo2geom_{step_value}_enriched"
    drop_sql = f"DROP TABLE IF EXISTS {table_name};"

    create_sql = f"""
    CREATE TABLE {table_name} AS
    WITH polygonized_edges AS (
        SELECT
            (ST_Dump(ST_Polygonize(e.geometry))).geom::geometry(Polygon, 28992) AS polygon_geom
        FROM yan_tgap_edge e
        WHERE e.step_low <= {step_value} AND e.step_high > {step_value}
    )
    SELECT
        p.polygon_geom,
        f.face_id,
        f.feature_class,
        ff.name
    FROM polygonized_edges p
    JOIN yan_tgap_face f
      ON ST_Contains(p.polygon_geom, f.pip_geometry)
    LEFT JOIN yan_face ff
      ON ff.face_id = f.face_id
    WHERE f.step_low <= {step_value} 
      AND f.step_high > {step_value};
    """

    with conn.cursor() as cur:
        cur.execute(drop_sql)
        cur.execute(create_sql)

    conn.commit()


def export_slice_to_gpkg(conn, step_value, out_gpkg):
    """
    Reads the slice table from PostGIS into a GeoDataFrame and writes it to .gpkg.
    """
    table_name = f"yan_topo2geom_{step_value}_enriched"
    sql = f"SELECT face_id, feature_class, name, polygon_geom FROM {table_name};"

    gdf = gpd.read_postgis(sql, engine, geom_col="polygon_geom")

    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)

    gdf.to_file(out_gpkg, layer="map_slice", driver="GPKG")
    print(f"Exported slice for step={step_value} to {out_gpkg}, layer=map_slice")


def generate_skeleton_labels(intermediate_gpkg, output_gpkg, do_simplify=True, simplify_tolerance=10.0):
    """
    Calls the genSkeleton's `generate_skeleton_for_gpkg()` function to do
    all skeleton-based logic and write labeled layers into output_gpkg.
    """
    if os.path.exists(output_gpkg):
        os.remove(output_gpkg)

    generate_skeleton_for_gpkg(
        input_gpkg=intermediate_gpkg,
        output_gpkg=output_gpkg,
        do_simplify=do_simplify,
        simplify_tolerance=simplify_tolerance
    )


def insert_labels_into_anchors_table(conn, step_value, gpkg_file):
    """
    Reads label layers (roads, water, building anchors) from gpkg_file
    and inserts them into label_anchors_from_slices table with the step_value column.
    """
    if not os.path.exists(gpkg_file):
        print(f"Warning: {gpkg_file} not found. No labels to insert.")
        return

    layers = fiona.listlayers(gpkg_file)
    if not layers:
        print(f"Warning: {gpkg_file} has no label layers.")
        return

    table_name = "label_anchors_from_slices"

    candidate_label_layers = [
        "map_slice_roads_labels",
        "map_slice_water_labels",
        "map_slice_buildings_centers",
    ]

    insert_sql = f"""
    INSERT INTO {table_name}
    (step_value, face_id, feature_class, name, anchor_geom, angle)
    VALUES (%s, %s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 28992), %s);
    """

    with conn.cursor() as cur:
        for lyr in candidate_label_layers:
            if lyr in layers:
                gdf = gpd.read_file(gpkg_file, layer=lyr)

                # Some skeleton code might store angle under 'rotation'
                angle_col = "angle" if "angle" in gdf.columns else "rotation"

                for _, row in gdf.iterrows():
                    face_id = row.get("face_id")
                    fclass = row.get("feature_class")
                    name = row.get("name")
                    angle = row.get(angle_col, 0.0)

                    geom_wkt = row["geometry"].wkt
                    cur.execute(insert_sql, (
                        step_value,
                        face_id,
                        fclass,
                        name,
                        geom_wkt,
                        angle
                    ))

    conn.commit()


################################################################################
# Main flow
################################################################################
def main(use_intermediate_files=False):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.autocommit = True

    # ------------------
    # 1) Create (or reset) anchors table
    # ------------------
    create_or_reset_anchors_table(conn)

    # ------------------
    # 2) Decide on base scale denominator
    # ------------------
    base_denominator = 10_000
    dataset_name = "yan"
    scale_step = ScaleStep(init_scale=base_denominator, topo_nm=dataset_name)

    # Example: gather some denominators by doubling from the base
    denominators = []
    current = base_denominator
    for _ in range(4):
        denominators.append(current)
        current *= 2

    print("Using denominators:", denominators)

    # Or a user-defined set, e.g.: denominators = [10000, 25000, 50000, 100000, ...]

    # ------------------
    # 3) For each denominator, calculate the step -> generate anchors -> store in table
    # ------------------
    for denom in denominators:
        step_val = scale_step.step_for_scale(denom)
        step_val = int(round(step_val, 0))

        print(f"\n--- Processing scale=1:{denom} => step={step_val} ---")

        if use_intermediate_files:
            # Step A: Create slice table in PostGIS for that step
            create_slice_table(conn, step_val)

            output_dir = "gpkg"
            os.makedirs(output_dir, exist_ok=True)

            # Step B: Export slice to .gpkg
            intermediate_gpkg = f"gpkg/slice_intermediate_{step_val}.gpkg"
            export_slice_to_gpkg(conn, step_val, out_gpkg=intermediate_gpkg)

            # Step C: Generate skeleton-labeled output
            skeleton_output_gpkg = f"gpkg/skeleton_output_{step_val}.gpkg"
            generate_skeleton_labels(intermediate_gpkg, skeleton_output_gpkg, do_simplify=True, simplify_tolerance=10.0)

            # Step D: Insert labels from skeleton output into label_anchors_from_slices
            insert_labels_into_anchors_table(conn, step_val, skeleton_output_gpkg)
        else:
            # Process geometries directly without intermediate files
            process_geometries_directly(conn, step_val, do_simplify=True, simplify_tolerance=10.0)

    print("All slices processed. Check label_anchors_from_slices for results.")
    conn.close()


if __name__ == "__main__":
    # Set use_intermediate_files=True if you want to use the original approach with intermediate files
    # Set use_intermediate_files=False to process geometries directly
    main(use_intermediate_files=False)
