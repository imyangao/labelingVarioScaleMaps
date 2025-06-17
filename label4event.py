import psycopg2
import shapely.wkb
from collections import defaultdict

from scalestep import ScaleStep
from labeling_core.db import get_connection, ROAD_MIN, ROAD_MAX, WATER_MIN, WATER_MAX, BULD_MIN, BULD_MAX
from labeling_core.anchors import compute_skeleton_anchors, compute_building_anchor
from labeling_core.traces import assign_label_trace_ids, compute_3d_bounding_boxes, visualize_3d_bounding_boxes

##############################################################################
# Retrieve faces of interest
##############################################################################
def get_faces_of_interest(conn):
    sql = f"""
        SELECT f.face_id,
               f.step_low,
               f.step_high,
               f.feature_class,
               y.name
        FROM newyan_tgap_face f
        LEFT JOIN newyan_face y
          ON y.face_id = f.face_id
        WHERE (
             (f.feature_class >= {ROAD_MIN} AND f.feature_class < {ROAD_MAX})
          OR (f.feature_class >= {WATER_MIN} AND f.feature_class < {WATER_MAX})
          OR (f.feature_class >= {BULD_MIN} AND f.feature_class < {BULD_MAX})
        )
        ORDER BY f.face_id;
        """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return rows

##############################################################################
# Retrieve edges bounding a given face
##############################################################################
def get_edges_for_face(conn, face_id, face_step_low, face_step_high):
    sql = """
    SELECT
      edge_id,
      step_low,
      step_high,
      start_node_id,
      end_node_id,
      left_face_id_low,
      right_face_id_low,
      left_face_id_high,
      right_face_id_high,
      edge_class,
      ST_AsBinary(geometry) AS geom_wkb
    FROM newyan_tgap_edge
    WHERE
      step_high > %s
      AND step_low < %s
      AND (
           left_face_id_low = %s
        OR right_face_id_low = %s
        OR left_face_id_high = %s
        OR right_face_id_high = %s
      );
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            face_step_low,
            face_step_high,
            face_id, face_id,
            face_id, face_id
        ))
        rows = cur.fetchall()
    return rows

##############################################################################
# Sub-interval logic
##############################################################################
def intervals_where_edge_bounds_face_noSwitch(face_id, f_low, f_high, edge_record):
    (edge_id,
     e_low, e_high,
     start_node_id, end_node_id,
     lf_low, rf_low,
     lf_high, rf_high,
     edge_class,
     geom_wkb) = edge_record

    overall_start = max(e_low, f_low)
    overall_end   = min(e_high, f_high)
    if overall_start >= overall_end:
        return

    if lf_low == face_id or rf_low == face_id:
        yield (overall_start, overall_end, geom_wkb)

##############################################################################
# Polygonize lines
##############################################################################
def polygonize_in_postgis_and_get_polygon(conn, line_wkbs):
    if not line_wkbs:
        return None

    with conn.cursor() as cur:
        sql = """
        WITH edges AS (
          SELECT ST_GeomFromWKB(unnest(%s), 28992) AS geom
        ),
        unioned AS (
          SELECT ST_Union(geom) AS union_geom
          FROM edges
        ),
        polys AS (
          SELECT ST_CollectionExtract(ST_Polygonize(unioned.union_geom), 3) AS poly
          FROM unioned
        )
        SELECT ST_AsBinary(ST_Union(poly)) AS fullpoly_wkb
        FROM polys
        WHERE NOT ST_IsEmpty(poly);
        """
        cur.execute(sql, (line_wkbs,))
        row = cur.fetchone()
        if not row or row[0] is None:
            return None

        wkb = row[0]
        poly_shp = shapely.wkb.loads(bytes(wkb))
        if poly_shp.is_empty:
            return None
        return poly_shp

##############################################################################
# Main pipeline
##############################################################################
def main(do_simplify=False, simplify_tolerance=1.0):
    conn = get_connection()

    # # Initialize ScaleStep
    # base_scale = 10000
    # dataset_name = 'newyan'
    # scale_step = ScaleStep(base_scale, dataset_name)

    # Create label_anchors table
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS label_anchors CASCADE;")
        cur.execute("""
        CREATE TABLE label_anchors (
            label_id   SERIAL PRIMARY KEY,
            face_id    INTEGER,
            step_value INTEGER,
            feature_class INTEGER,
            name TEXT,
            anchor_geom geometry(POINT, 28992),
            face_geom    geometry(MULTIPOLYGON, 28992),
            angle      DOUBLE PRECISION
        );
        """)

    # Get faces
    faces = get_faces_of_interest(conn)
    print(f"Found {len(faces)} faces of interest.")

    # Dictionary to track previous anchors for each face
    # Format: {face_id: {step_value: [(point, angle), ...]}}
    previous_anchors = {}

    # Each row: (face_id, step_low, step_high, feature_class, name)
    for (face_id, f_low, f_high, fclass, face_name) in faces:
        edges = get_edges_for_face(conn, face_id, f_low, f_high)
        if not edges:
            continue

        edge_intervals = []
        event_steps = {f_low, f_high}

        for e in edges:
            for (start, end, line_wkb) in intervals_where_edge_bounds_face_noSwitch(
                face_id, f_low, f_high, e
            ):
                edge_intervals.append((start, end, line_wkb))
                event_steps.add(start)
                event_steps.add(end)

        sorted_events = sorted(event_steps)

        for i in range(len(sorted_events) - 1):
            S = sorted_events[i]
            T = sorted_events[i + 1]
            if S == T:
                continue
            if S >= f_high:
                continue

            boundary_wkbs = [
                line_wkb for (start, end, line_wkb) in edge_intervals
                if start <= S < end
            ]

            if not boundary_wkbs:
                continue

            poly_shp = polygonize_in_postgis_and_get_polygon(conn, boundary_wkbs)
            if not poly_shp or poly_shp.is_empty:
                continue
            poly_wkt = poly_shp.wkt

            # Decide how to compute anchors
            if (ROAD_MIN <= fclass < ROAD_MAX) or (WATER_MIN <= fclass < WATER_MAX):
                # roads/water
                anchors = compute_skeleton_anchors(
                    polygon=poly_shp,
                    do_simplify=do_simplify,
                    simplify_tolerance=simplify_tolerance
                )
                if not anchors:
                    c = poly_shp.centroid
                    anchors = [(c, 0.0)]
            elif BULD_MIN <= fclass < BULD_MAX:
                # building
                anchors = compute_building_anchor(
                    polygon=poly_shp
                )
                if not anchors:
                    a_pt = poly_shp.centroid
                    a_angle = 0.0
                    anchors = [(a_pt, a_angle)]
            else:
                # fallback => centroid
                c = poly_shp.centroid
                anchors = [(c, 0.0)]

            # Find the previous step with anchors for this face
            prev_step = None
            prev_anchors = None
            if face_id in previous_anchors:
                # Find the highest step less than current step that has anchors
                for step in sorted(previous_anchors[face_id].keys(), reverse=True):
                    if step < S and previous_anchors[face_id][step]:
                        prev_step = step
                        prev_anchors = previous_anchors[face_id][step]
                        break

            # If we have previous anchors, limit current anchors based on proximity
            if prev_anchors and len(anchors) > len(prev_anchors):
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
                
                # Keep only the closest anchors
                anchors = [anchor for anchor, _ in anchor_distances[:len(prev_anchors)]]

            # Store current anchors for future reference
            if face_id not in previous_anchors:
                previous_anchors[face_id] = {}
            previous_anchors[face_id][S] = anchors

            # Insert anchors
            for (anchor_pt, angle) in anchors:
                wkt = anchor_pt.wkt
                # Insert into database
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO label_anchors(face_id, step_value, feature_class, name, anchor_geom, face_geom, angle)
                        VALUES (%s, %s, %s, %s, ST_GeomFromText(%s, 28992), ST_Multi(ST_GeomFromText(%s, 28992)), %s)
                    """, (face_id, S, fclass, face_name, wkt, poly_wkt, angle))

    print("Done! Label anchors inserted.")

    # Done inserting; now assign label_trace_id
    assign_label_trace_ids(conn, table_name='label_anchors', distance_per_step=float('inf'))
    print("Done! label_trace_id assigned.")

    # # Compute 3D bounding boxes and visualize
    # bounds_table_name = 'label_trace_3d_bounds'
    # bounding_boxes = compute_3d_bounding_boxes(conn, 'label_anchors', bounds_table_name, create_table=True)
    # print(f"Done! 3D bounding boxes computed and stored in {bounds_table_name}.")
    #
    # # Visualize them in 3D
    # visualize_3d_bounding_boxes(bounding_boxes)

    conn.close()


if __name__ == "__main__":
    main(do_simplify=False, simplify_tolerance=0.0)

