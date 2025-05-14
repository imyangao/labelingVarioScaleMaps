import psycopg2
import os
import fiona
import geopandas as gpd
import math
from shapely.geometry import Point, LineString
from shapely.algorithms.polylabel import polylabel
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

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
    # Dictionary to track previous anchors for each face
    # Format: {face_id: {step_value: [(point, angle), ...]}}
    # Make it global so it persists between calls
    global previous_anchors
    if 'previous_anchors' not in globals():
        previous_anchors = {}

    print(f"\nProcessing step {step_value}")

    # Query to get all faces at the specified step
    sql = f"""
    WITH polygonized_edges AS (
        SELECT
            (ST_Dump(ST_Polygonize(e.geometry))).geom::geometry(Polygon, 28992) AS polygon_geom
        FROM newyan_tgap_edge e
        WHERE e.step_low <= {step_value} AND e.step_high > {step_value}
    )
    SELECT
        p.polygon_geom,
        f.face_id,
        f.feature_class,
        ff.name
    FROM polygonized_edges p
    JOIN newyan_tgap_face f
      ON ST_Contains(p.polygon_geom, f.pip_geometry)
    LEFT JOIN newyan_face ff
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
                anchors = process_road_geometry(conn, step_value, polygon, feature_class, face_id, name, 
                                     do_simplify, simplify_tolerance)
            elif 12000 <= feature_class < 13000:  # Water
                anchors = process_water_geometry(conn, step_value, polygon, feature_class, face_id, name, 
                                       do_simplify, simplify_tolerance)
            elif 13000 <= feature_class < 14000:  # Buildings
                anchors = process_building_geometry(conn, step_value, polygon, feature_class, face_id, name)
            else:
                anchors = None

            if anchors:
                # # Debug: Print current anchor count
                # print(f"Face {face_id} at step {step_value}: {len(anchors)} anchors")

                # Find the previous step with anchors for this face
                prev_step = None
                prev_anchors = None
                if face_id in previous_anchors:
                    # Find the highest step less than current step that has anchors
                    for step in sorted(previous_anchors[face_id].keys(), reverse=True):
                        if step < step_value and previous_anchors[face_id][step]:
                            prev_step = step
                            prev_anchors = previous_anchors[face_id][step]
                            # print(f"  Previous step {prev_step} had {len(prev_anchors)} anchors")
                            break

                # If we have previous anchors and current anchors exceed the previous count
                if prev_anchors and len(anchors) > len(prev_anchors):
                    print(f"  Reducing from {len(anchors)} to {len(prev_anchors)} anchors")
                    # Create a list of (current_anchor, min_distance_to_prev_anchors)
                    anchor_distances = []
                    for curr_anchor in anchors:
                        min_dist = min(
                            curr_anchor[0].distance(prev_anchor[0])
                            for prev_anchor in prev_anchors
                        )
                        anchor_distances.append((curr_anchor, min_dist))
                    
                    # Sort by minimum distance to previous anchors
                    anchor_distances.sort(key=lambda x: x[1])
                    
                    # Keep only the closest anchors, not exceeding the previous count
                    anchors = [anchor for anchor, _ in anchor_distances[:len(prev_anchors)]]

                # Store current anchors for future reference
                if face_id not in previous_anchors:
                    previous_anchors[face_id] = {}
                previous_anchors[face_id][step_value] = anchors

                # Insert anchors into database
                insert_sql = """
                INSERT INTO label_anchors_from_slices
                (step_value, face_id, feature_class, name, anchor_geom, angle)
                VALUES (%s, %s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 28992), %s);
                """
                
                with conn.cursor() as cur:
                    for anchor_pt, angle in anchors:
                        cur.execute(insert_sql, (
                            step_value,
                            face_id,
                            feature_class,
                            name,
                            anchor_pt.wkt,
                            angle
                        ))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error processing geometry at index {idx}: {e}")


def process_road_geometry(conn, step_value, polygon, feature_class, face_id, name, 
                         do_simplify, simplify_tolerance):
    """Process road geometry and return anchor points."""
    # Optional polygon simplification
    if do_simplify and simplify_tolerance > 0:
        poly_for_skel = polygon.simplify(simplify_tolerance, preserve_topology=True)
    else:
        poly_for_skel = polygon

    # Build raw skeleton
    raw_skel_lines = build_skeleton_lines(poly_for_skel)
    if not raw_skel_lines:
        return None

    # Convert to graph -> find junctions -> connect them
    G = lines_to_graph(raw_skel_lines)
    junctions = get_junction_nodes(G, min_degree=3)
    if len(junctions) < 2:
        # not enough skeleton complexity
        return None

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

    anchors = []
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

        anchors.append((midpoint, angle_deg))

    return anchors


def process_water_geometry(conn, step_value, polygon, feature_class, face_id, name, 
                          do_simplify, simplify_tolerance):
    """Process water geometry and return anchor points."""
    # Optional polygon simplification
    if do_simplify and simplify_tolerance > 0:
        poly_for_skel = polygon.simplify(simplify_tolerance, preserve_topology=True)
    else:
        poly_for_skel = polygon

    # Build raw skeleton
    raw_skel_lines = build_skeleton_lines(poly_for_skel)
    if not raw_skel_lines:
        return None

    # Convert to graph -> find junctions -> connect them
    G = lines_to_graph(raw_skel_lines)
    junctions = get_junction_nodes(G, min_degree=3)
    if len(junctions) < 2:
        return None

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

    anchors = []
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

        anchors.append((midpoint, angle_deg))

    return anchors


def process_building_geometry(conn, step_value, polygon, feature_class, face_id, name):
    """Process building geometry and return anchor points."""
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

        return [(center, best_angle)]
    return None


################################################################################
# Optional: Create intermediate files if needed
################################################################################
def create_slice_table(conn, step_value):
    """
    Creates a slice table for the specified step_value
    """
    table_name = f"newyan_topo2geom_{step_value}_enriched"
    drop_sql = f"DROP TABLE IF EXISTS {table_name};"

    create_sql = f"""
    CREATE TABLE {table_name} AS
    WITH polygonized_edges AS (
        SELECT
            (ST_Dump(ST_Polygonize(e.geometry))).geom::geometry(Polygon, 28992) AS polygon_geom
        FROM newyan_tgap_edge e
        WHERE e.step_low <= {step_value} AND e.step_high > {step_value}
    )
    SELECT
        p.polygon_geom,
        f.face_id,
        f.feature_class,
        ff.name
    FROM polygonized_edges p
    JOIN newyan_tgap_face f
      ON ST_Contains(p.polygon_geom, f.pip_geometry)
    LEFT JOIN newyan_face ff
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
    table_name = f"newyan_topo2geom_{step_value}_enriched"
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

    # Dictionary to track previous anchors for each face
    # Format: {face_id: {step_value: [(point, angle), ...]}}
    global previous_anchors
    if 'previous_anchors' not in globals():
        previous_anchors = {}

    with conn.cursor() as cur:
        for lyr in candidate_label_layers:
            if lyr in layers:
                gdf = gpd.read_file(gpkg_file, layer=lyr)

                # Some skeleton code might store angle under 'rotation'
                angle_col = "angle" if "angle" in gdf.columns else "rotation"

                # Group anchors by face_id
                face_anchors = defaultdict(list)
                for _, row in gdf.iterrows():
                    face_id = row.get("face_id")
                    fclass = row.get("feature_class")
                    name = row.get("name")
                    angle = row.get(angle_col, 0.0)
                    geom = row["geometry"]
                    face_anchors[face_id].append((geom, angle, fclass, name))

                # Process each face's anchors
                for face_id, anchors in face_anchors.items():
                    # Find the previous step with anchors for this face
                    prev_step = None
                    prev_anchors = None
                    if face_id in previous_anchors:
                        # Find the highest step less than current step that has anchors
                        for step in sorted(previous_anchors[face_id].keys(), reverse=True):
                            if step < step_value and previous_anchors[face_id][step]:
                                prev_step = step
                                prev_anchors = previous_anchors[face_id][step]
                                break

                    # If we have previous anchors and current anchors exceed the previous count
                    if prev_anchors and len(anchors) > len(prev_anchors):
                        print(f"  Reducing from {len(anchors)} to {len(prev_anchors)} anchors for face {face_id}")
                        # Create a list of (current_anchor, min_distance_to_prev_anchors)
                        anchor_distances = []
                        for curr_anchor in anchors:
                            min_dist = min(
                                curr_anchor[0].distance(prev_anchor[0])
                                for prev_anchor in prev_anchors
                            )
                            anchor_distances.append((curr_anchor, min_dist))
                        
                        # Sort by minimum distance to previous anchors
                        anchor_distances.sort(key=lambda x: x[1])
                        
                        # Keep only the closest anchors, not exceeding the previous count
                        anchors = [anchor for anchor, _ in anchor_distances[:len(prev_anchors)]]

                    # Store current anchors for future reference
                    if face_id not in previous_anchors:
                        previous_anchors[face_id] = {}
                    previous_anchors[face_id][step_value] = [(geom, angle) for geom, angle, _, _ in anchors]

                    # Insert the anchors into the database
                    for geom, angle, fclass, name in anchors:
                        geom_wkt = geom.wkt
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
# Trace anchor points across slices
################################################################################
def assign_label_trace_ids(conn, distance_threshold=50.0):
    """
    Post-process the label_anchors table to group anchors from the same face
    that persist across different steps into the same label_trace_id.

    distance_threshold is how close two anchors must be (in map units)
    to be considered the 'same' anchor in consecutive steps.
    """
    from scipy.optimize import linear_sum_assignment
    import math
    from collections import defaultdict

    # 1) Add new column if it doesn't exist
    with conn.cursor() as cur:
        try:
            cur.execute("ALTER TABLE label_anchors_from_slices ADD COLUMN label_trace_id BIGSERIAL;")
        except psycopg2.errors.DuplicateColumn:
            conn.rollback()

    # 2) Fetch all anchors
    with conn.cursor() as cur:
        cur.execute("""
            SELECT label_id, face_id, step_value,
                   ST_X(anchor_geom) AS x,
                   ST_Y(anchor_geom) AS y
            FROM label_anchors_from_slices
            ORDER BY face_id, step_value;
        """)
        rows = cur.fetchall()

    # Group by face_id
    faces_map = defaultdict(list)
    for (lbl_id, face_id, step_val, x, y) in rows:
        faces_map[face_id].append((lbl_id, step_val, x, y))

    # We'll store label_id -> label_trace_id
    assignments = {}
    next_trace_id = 1

    for face_id, anchors in faces_map.items():
        # Sort by ascending step_value
        anchors.sort(key=lambda r: r[1])

        # Split by step_value
        from itertools import groupby
        steps_data = [(step_val, list(grp))
                      for step_val, grp in groupby(anchors, key=lambda r: r[1])]

        # This will hold (trace_id, x, y) for anchors from the *previous* step
        active_traces = []

        for idx, (step_val, anchor_list) in enumerate(steps_data):
            if idx == 0:
                # All anchors in the first step get new trace IDs
                for (lbl_id, stv, x, y) in anchor_list:
                    assignments[lbl_id] = next_trace_id
                    active_traces.append((next_trace_id, x, y))
                    next_trace_id += 1
            else:
                # Since we know previous step has more or equal anchors,
                # we can directly match current anchors to their closest previous anchors
                new_active = []
                matched_prev_indices = set()

                # For each current anchor, find the closest previous anchor
                for (lbl_id, stv, x, y) in anchor_list:
                    min_dist = float('inf')
                    best_prev_idx = -1

                    # Find closest previous anchor
                    for i, (t_id, px, py) in enumerate(active_traces):
                        if i in matched_prev_indices:
                            continue
                        dist = math.hypot(x - px, y - py)
                        if dist < min_dist and dist <= distance_threshold:
                            min_dist = dist
                            best_prev_idx = i

                    if best_prev_idx != -1:
                        # Match found - use the same trace_id
                        trace_id = active_traces[best_prev_idx][0]
                        assignments[lbl_id] = trace_id
                        matched_prev_indices.add(best_prev_idx)
                        new_active.append((trace_id, x, y))
                    else:
                        # No match found - create new trace_id
                        assignments[lbl_id] = next_trace_id
                        new_active.append((next_trace_id, x, y))
                        next_trace_id += 1

                # Update active_traces for next iteration
                active_traces = new_active
    # 3) Update the database
    with conn.cursor() as cur:
        for lbl_id, trace_id in assignments.items():
            cur.execute("""
                UPDATE label_anchors_from_slices
                   SET label_trace_id = %s
                 WHERE label_id = %s;
            """, (trace_id, lbl_id))

    conn.commit()
    print("label_trace_id assigned to all anchors.")

def compute_3d_bounding_boxes(conn, create_table=True):
    """
    For each label_trace_id, compute an axis-aligned bounding box in x, y,
    and step_value space. Then optionally store them in a new table for
    further use or visualization.
    """
    if create_table:
        with conn.cursor() as cur:
            # Drop if exists for demonstration; remove in real environment
            cur.execute("DROP TABLE IF EXISTS label_trace_3d_bounds_slices;")
            # We will store minX, maxX, minY, maxY, minStep, maxStep
            cur.execute("""
                CREATE TABLE label_trace_3d_bounds_slices (
                    label_trace_id BIGINT,
                    min_x DOUBLE PRECISION,
                    max_x DOUBLE PRECISION,
                    min_y DOUBLE PRECISION,
                    max_y DOUBLE PRECISION,
                    min_step INTEGER,
                    max_step INTEGER
                );
            """)
        conn.commit()

    # Fetch anchors grouped by label_trace_id
    with conn.cursor() as cur:
        cur.execute("""
            SELECT label_trace_id,
                   ST_X(anchor_geom) as x,
                   ST_Y(anchor_geom) as y,
                   step_value
            FROM label_anchors_from_slices
            WHERE label_trace_id IS NOT NULL
            ORDER BY label_trace_id;
        """)
        rows = cur.fetchall()

    # Dictionary: trace_id -> list of (x, y, step)
    trace_map = defaultdict(list)
    for t_id, x, y, stp in rows:
        trace_map[t_id].append((x, y, stp))

    # Compute bounding boxes
    bounding_boxes = {}
    for t_id, coords in trace_map.items():
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        steps = [c[2] for c in coords]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        min_stp = min(steps)
        max_stp = max(steps)
        bounding_boxes[t_id] = (min_x, max_x, min_y, max_y, min_stp, max_stp)

    # Insert bounding boxes into table
    if create_table:
        with conn.cursor() as cur:
            for t_id, (mnx, mxx, mny, mxy, mns, mxs) in bounding_boxes.items():
                cur.execute("""
                    INSERT INTO label_trace_3d_bounds_slices (
                        label_trace_id,
                        min_x, max_x, min_y, max_y,
                        min_step, max_step
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, (t_id, mnx, mxx, mny, mxy, mns, mxs))
        conn.commit()

    return bounding_boxes


def visualize_3d_bounding_boxes(bounding_boxes):
    """
    Create a very simple 3D plot of each axis-aligned bounding box using Matplotlib.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Step')

    for t_id, (min_x, max_x, min_y, max_y, min_step, max_step) in bounding_boxes.items():
        # "Corners" of the 3D box
        corners = [
            (min_x, min_y, min_step),
            (min_x, min_y, max_step),
            (min_x, max_y, min_step),
            (min_x, max_y, max_step),
            (max_x, min_y, min_step),
            (max_x, min_y, max_step),
            (max_x, max_y, min_step),
            (max_x, max_y, max_step),
        ]
        # We'll draw line segments between these corners
        # For an axis-aligned box, we have 12 edges:
        edges = [
            (0,1), (0,2), (0,4), (7,3), (7,5), (7,6), (1,3), (1,5), (2,3), (2,6), (4,5), (4,6)
        ]
        for (i, j) in edges:
            p1 = corners[i]
            p2 = corners[j]
            xs = [p1[0], p2[0]]
            ys = [p1[1], p2[1]]
            zs = [p1[2], p2[2]]
            ax.plot(xs, ys, zs)

    plt.title("3D Bounding Boxes by label_trace_id")
    plt.show()


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
    base_denominator = 10000
    dataset_name = "newyan"
    scale_step = ScaleStep(init_scale=base_denominator, topo_nm=dataset_name)

    # Example: gather some denominators by doubling from the base
    denominators = []
    current = base_denominator
    for _ in range(4):
        denominators.append(current)
        current *= 2

    print("Using denominators:", denominators)

    # ------------------
    # 3) For each denominator, calculate the step -> generate anchors -> store in table
    # ------------------
    # Process steps in ascending order to ensure proper anchor tracking
    for denom in sorted(denominators):
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
            generate_skeleton_labels(intermediate_gpkg, skeleton_output_gpkg, do_simplify=True, simplify_tolerance=1.0)

            # Step D: Insert labels from skeleton output into label_anchors_from_slices
            insert_labels_into_anchors_table(conn, step_val, skeleton_output_gpkg)
        else:
            # Process geometries directly without intermediate files
            process_geometries_directly(conn, step_val, do_simplify=True, simplify_tolerance=1.0)

    # ------------------
    # 4) Trace anchor points across slices
    # ------------------
    print("\n--- Tracing anchor points across slices ---")
    assign_label_trace_ids(conn, distance_threshold=50.0)
    
    # ------------------
    # 5) Compute and visualize 3D bounding boxes
    # ------------------
    print("\n--- Computing 3D bounding boxes ---")
    bounding_boxes = compute_3d_bounding_boxes(conn, create_table=True)

    # Visualize the bounding boxes
    print("\n--- Visualizing 3D bounding boxes ---")
    visualize_3d_bounding_boxes(bounding_boxes)

    print("All slices processed. Check label_anchors_from_slices for results.")
    conn.close()


if __name__ == "__main__":
    # Set use_intermediate_files=True if you want to use the original approach with intermediate files
    # Set use_intermediate_files=False to process geometries directly
    main(use_intermediate_files=False)
