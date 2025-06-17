import psycopg2
import os
import fiona
import geopandas as gpd
from collections import defaultdict

from scalestep import ScaleStep
from labeling_core.db import get_connection, get_engine
from labeling_core.skeleton import generate_skeleton_for_gpkg
from labeling_core.anchors import compute_skeleton_anchors, compute_building_anchor
from labeling_core.traces import assign_label_trace_ids, compute_3d_bounding_boxes, visualize_3d_bounding_boxes


################################################################################
# Database & table setup
################################################################################

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
def process_geometries_directly(conn, step_value, do_simplify=False, simplify_tolerance=10.0):
    """
    Process geometries directly from the database for a given step_value,
    compute anchor points based on feature_class, and insert them into the anchors table.
    """
    global previous_anchors
    if 'previous_anchors' not in globals():
        previous_anchors = {}

    print(f"\nProcessing step {step_value}")
    engine = get_engine()

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
    
    gdf = gpd.read_postgis(sql, engine, geom_col="polygon_geom")
    
    if gdf.empty:
        print(f"No data found for step_value {step_value}")
        return
    
    for idx, row in gdf.iterrows():
        try:
            # Get values that might be missing or NaN
            feature_class_val = row.get('feature_class', None)
            face_id_val = row.get('face_id', None)

            # Use pandas isna to check for both None and NaN which can't be converted to int.
            if gpd.pd.isna(feature_class_val) or gpd.pd.isna(face_id_val):
                if gpd.pd.isna(face_id_val):
                    print(f"Warning: Skipping geometry at index {idx} due to missing face_id.")
                # We can silently skip if feature_class is missing, as it's not a critical error.
                continue
            
            # It is now safe to cast to int
            polygon = row['polygon_geom'].buffer(0)
            feature_class = int(feature_class_val)
            face_id = int(face_id_val)
            name = row.get('name', None)
                
            anchors = []
            if 10000 <= feature_class < 11000 or 12000 <= feature_class < 13000:  # Roads or Water
                anchors = compute_skeleton_anchors(polygon, do_simplify, simplify_tolerance)
            elif 13000 <= feature_class < 14000:  # Buildings
                anchors = compute_building_anchor(polygon)
            
            # if not anchors:
            #     anchors = [(polygon.centroid, 0.0)]

            if anchors:
                prev_anchors_list = []
                if face_id in previous_anchors:
                    for step in sorted(previous_anchors.get(face_id, {}).keys(), reverse=True):
                        if step < step_value and previous_anchors[face_id][step]:
                            prev_anchors_list = previous_anchors[face_id][step]
                            break

                if prev_anchors_list and len(anchors) > len(prev_anchors_list):
                    anchor_distances = []
                    for curr_anchor in anchors:
                        min_dist = min(curr_anchor[0].distance(prev_anchor[0]) for prev_anchor in prev_anchors_list)
                        anchor_distances.append((curr_anchor, min_dist))
                    
                    anchor_distances.sort(key=lambda x: x[1])
                    anchors = [anchor for anchor, _ in anchor_distances[:len(prev_anchors_list)]]

                if face_id not in previous_anchors:
                    previous_anchors[face_id] = {}
                previous_anchors[face_id][step_value] = anchors

                with conn.cursor() as cur:
                    for anchor_pt, angle in anchors:
                        cur.execute("""
                            INSERT INTO label_anchors_from_slices
                            (step_value, face_id, feature_class, name, anchor_geom, angle)
                            VALUES (%s, %s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 28992), %s);
                        """, (step_value, face_id, feature_class, name, anchor_pt.wkt, angle))
                conn.commit()
                
        except Exception as e:
            face_id_for_log = row.get('face_id', 'N/A')
            print(f"Error processing geometry at index {idx} for face {face_id_for_log}: {e}")


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
    engine = get_engine()
    gdf = gpd.read_postgis(sql, engine, geom_col="polygon_geom")
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)
    gdf.to_file(out_gpkg, layer="map_slice", driver="GPKG")
    print(f"Exported slice for step={step_value} to {out_gpkg}, layer=map_slice")


def insert_labels_into_anchors_table(conn, step_value, gpkg_file):
    """
    Reads label layers from gpkg_file and inserts them into anchors table.
    """
    global previous_anchors
    if 'previous_anchors' not in globals():
        previous_anchors = {}

    if not os.path.exists(gpkg_file):
        print(f"Warning: {gpkg_file} not found. No labels to insert.")
        return

    layers = fiona.listlayers(gpkg_file)
    if not layers:
        print(f"Warning: {gpkg_file} has no label layers.")
        return

    candidate_label_layers = ["map_slice_roads_labels", "map_slice_water_labels", "map_slice_buildings_centers"]
    
    with conn.cursor() as cur:
        for lyr in candidate_label_layers:
            if lyr in layers:
                gdf = gpd.read_file(gpkg_file, layer=lyr)
                angle_col = "angle" if "angle" in gdf.columns else "rotation"

                face_anchors = defaultdict(list)
                for _, row in gdf.iterrows():
                    face_id = row.get("face_id")
                    fclass = row.get("feature_class")
                    name = row.get("name")
                    angle = row.get(angle_col, 0.0)
                    geom = row["geometry"]
                    face_anchors[face_id].append({"geom": geom, "angle": angle, "fclass": fclass, "name": name})

                for face_id, anchors_data in face_anchors.items():
                    anchors = [(d["geom"], d["angle"]) for d in anchors_data]
                    
                    prev_anchors_list = []
                    if face_id in previous_anchors:
                        for step in sorted(previous_anchors.get(face_id, {}).keys(), reverse=True):
                            if step < step_value and previous_anchors[face_id][step]:
                                prev_anchors_list = previous_anchors[face_id][step]
                                break

                    if prev_anchors_list and len(anchors) > len(prev_anchors_list):
                        anchor_distances = []
                        for curr_anchor in anchors:
                            min_dist = min(curr_anchor[0].distance(prev_anchor[0]) for prev_anchor in prev_anchors_list)
                            anchor_distances.append((curr_anchor, min_dist))
                        
                        anchor_distances.sort(key=lambda x: x[1])
                        anchors = [anchor for anchor, _ in anchor_distances[:len(prev_anchors_list)]]

                    if face_id not in previous_anchors:
                        previous_anchors[face_id] = {}
                    previous_anchors[face_id][step_value] = anchors

                    for i, (geom, angle) in enumerate(anchors):
                        anchor_info = next((d for d in anchors_data if d["geom"] == geom), None)
                        if anchor_info:
                            cur.execute("""
                                INSERT INTO label_anchors_from_slices
                                (step_value, face_id, feature_class, name, anchor_geom, angle)
                                VALUES (%s, %s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 28992), %s);
                            """, (step_value, face_id, anchor_info["fclass"], anchor_info["name"], geom.wkt, angle))
    conn.commit()


################################################################################
# Main flow
################################################################################
def main(use_intermediate_files=False):
    conn = get_connection()
    create_or_reset_anchors_table(conn)

    base_denominator = 10000
    dataset_name = "newyan"
    scale_step_calc = ScaleStep(init_scale=base_denominator, topo_nm=dataset_name)
    
    denominators = [base_denominator * (2**i) for i in range(4)]
    print("Using denominators:", denominators)

    for denom in sorted(denominators):
        step_val = int(round(scale_step_calc.step_for_scale(denom), 0))
        print(f"\n--- Processing scale=1:{denom} => step={step_val} ---")

        if use_intermediate_files:
            output_dir = "gpkg"
            os.makedirs(output_dir, exist_ok=True)
            intermediate_gpkg = os.path.join(output_dir, f"slice_intermediate_{step_val}.gpkg")
            skeleton_output_gpkg = os.path.join(output_dir, f"skeleton_output_{step_val}.gpkg")

            create_slice_table(conn, step_val)
            export_slice_to_gpkg(conn, step_val, out_gpkg=intermediate_gpkg)
            generate_skeleton_for_gpkg(intermediate_gpkg, skeleton_output_gpkg, do_simplify=False, simplify_tolerance=0.0)
            insert_labels_into_anchors_table(conn, step_val, skeleton_output_gpkg)
        else:
            process_geometries_directly(conn, step_val, do_simplify=False, simplify_tolerance=0.0)

    print("\n--- Tracing anchor points across slices ---")
    assign_label_trace_ids(conn, 'label_anchors_from_slices', distance_per_step=float('inf'))
    
    # print("\n--- Computing and visualizing 3D bounding boxes ---")
    # bounds_table = 'label_trace_3d_bounds_slices'
    # bounding_boxes = compute_3d_bounding_boxes(conn, 'label_anchors_from_slices', bounds_table, create_table=True)
    # visualize_3d_bounding_boxes(bounding_boxes)

    print("\nAll slices processed. Check 'label_anchors_from_slices' for results.")
    conn.close()


if __name__ == "__main__":
    main(use_intermediate_files=False)
