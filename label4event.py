import psycopg2
import shapely.wkb
from shapely.geometry import Point, LineString
from shapely.algorithms.polylabel import polylabel
import math
import geopandas as gpd
import os
from shapely.geometry import Polygon
from shapely.affinity import rotate
from scalestep import ScaleStep
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from collections import defaultdict

from genSkeleton import (
    build_skeleton_lines,
    lines_to_graph,
    get_junction_nodes,
    find_junction_to_junction_paths,
    merge_collinear_lines,
    largest_inscribed_circle,
)

##############################################################################
# Database connection
##############################################################################
DB_NAME = "tgap_test"
DB_USER = "postgres"
DB_PASS = "Gy@001130"
DB_HOST = "localhost"
DB_PORT = 5432

##############################################################################
# Feature-class ranges
##############################################################################
ROAD_MIN, ROAD_MAX = 10000, 11000
WATER_MIN, WATER_MAX = 12000, 13000
BULD_MIN, BULD_MAX = 13000, 14000

##############################################################################
# 1) Retrieve faces of interest
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
# 2) Retrieve edges bounding a given face
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
# 3) Sub-interval logic
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
# 4) Polygonize lines
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
# 5) Angle
##############################################################################
def compute_line_angle(line):
    coords = list(line.coords)
    (x1, y1), (x2, y2) = coords[0], coords[-1]
    dx, dy = (x2 - x1), (y2 - y1)
    angle_deg = math.degrees(math.atan2(dy, dx))
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
    return angle_deg

##############################################################################
# 6) Skeleton-based anchors
##############################################################################
def compute_skeleton_anchors(polygon, do_simplify=False, simplify_tolerance=1.0):
    if do_simplify and simplify_tolerance > 0:
        # preserve_topology=True helps avoid weird self-intersections
        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

    raw_skel_lines = build_skeleton_lines(polygon)
    if not raw_skel_lines:
        return []

    G = lines_to_graph(raw_skel_lines)
    junctions = get_junction_nodes(G, min_degree=3)
    if len(junctions) < 2:
        # fallback: just pick raw lines above threshold
        sorted_raw = sorted(raw_skel_lines, key=lambda ln: ln.length, reverse=True)
        if not sorted_raw:
            return []
        max_len = sorted_raw[0].length
        threshold_fraction = 0.25
        length_threshold = threshold_fraction * max_len

        out_anchors = []
        for ln in sorted_raw:
            if ln.length < length_threshold:
                break
            midpt = ln.interpolate(0.5, normalized=True)
            angle = compute_line_angle(ln)
            out_anchors.append((midpt, angle))
        return out_anchors

    # Real skeleton lines
    primary_paths = find_junction_to_junction_paths(G, junctions)
    merged_primary = merge_collinear_lines(primary_paths, angle_threshold=5.0)
    if not merged_primary:
        return []

    merged_primary.sort(key=lambda l: l.length, reverse=True)
    max_length = merged_primary[0].length
    length_threshold = 0.25 * max_length

    anchors = []
    for line in merged_primary:
        if line.length < length_threshold:
            break
        midpoint = line.interpolate(0.5, normalized=True)
        angle_deg = compute_line_angle(line)
        anchors.append((midpoint, angle_deg))

    return anchors

##############################################################################
# 7) Buildings: polylabel + orientation
##############################################################################
def compute_building_anchor(polygon):

    from shapely.algorithms.polylabel import polylabel

    if polygon.is_empty:
        return None, 0.0

    anchor = polylabel(polygon, tolerance=1.0)

    minrect = polygon.minimum_rotated_rectangle
    coords = list(minrect.exterior.coords)
    if len(coords) < 2:
        return anchor, 0.0

    best_len = 0.0
    best_angle = 0.0
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i+1]
        seg_len = Point(p1).distance(Point(p2))
        if seg_len > best_len:
            best_len = seg_len
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle_deg = math.degrees(math.atan2(dy, dx))
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180
            best_angle = angle_deg

    return anchor, best_angle


##############################################################################
# Assign label_trace_id
##############################################################################
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
            cur.execute("ALTER TABLE label_anchors ADD COLUMN label_trace_id BIGSERIAL;")
        except psycopg2.errors.DuplicateColumn:
            conn.rollback()

    # 2) Fetch all anchors
    with conn.cursor() as cur:
        cur.execute("""
            SELECT label_id, face_id, step_value,
                   ST_X(anchor_geom) AS x,
                   ST_Y(anchor_geom) AS y
            FROM label_anchors
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
                UPDATE label_anchors
                   SET label_trace_id = %s
                 WHERE label_id = %s;
            """, (trace_id, lbl_id))

    conn.commit()
    print("label_trace_id assigned to all anchors.")


##############################################################################
# Compute 3D bounding boxes
##############################################################################
def compute_3d_bounding_boxes(conn, create_table=True):
    """
    For each label_trace_id, compute an axis-aligned bounding box in x, y,
    and step_value space. Then optionally store them in a new table for
    further use or visualization.
    """
    if create_table:
        with conn.cursor() as cur:
            # Drop if exists for demonstration; remove in real environment
            cur.execute("DROP TABLE IF EXISTS label_trace_3d_bounds;")
            # We will store minX, maxX, minY, maxY, minStep, maxStep
            cur.execute("""
                CREATE TABLE label_trace_3d_bounds (
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
            FROM label_anchors
            WHERE label_trace_id IS NOT NULL AND fits is TRUE
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
                    INSERT INTO label_trace_3d_bounds (
                        label_trace_id,
                        min_x, max_x, min_y, max_y,
                        min_step, max_step
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, (t_id, mnx, mxx, mny, mxy, mns, mxs))
        conn.commit()

    # print("3D bounding boxes have been computed.")
    return bounding_boxes


##############################################################################
# Simple 3D visualization for bounding boxes
##############################################################################
def visualize_3d_bounding_boxes(bounding_boxes):
    """
    Create a very simple 3D plot of each axis-aligned bounding box using Matplotlib.
    """
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


def commit_to_first_anchor_only(conn):
    """Implements Method 1 (SQL post‑processing).

    For every *label_trace_id* that ever goes `fits = FALSE`, force every later step to
    be *FALSE* as well so that visibility becomes a monotone non‑increasing function of
    *step_value*.
    """
    sql = """
    WITH first_bad AS (
        SELECT label_trace_id,
               MIN(step_value) AS fail_at
        FROM   label_anchors
        WHERE  fits = FALSE
        GROUP  BY label_trace_id
    )
    UPDATE label_anchors AS la
    SET    fits = FALSE
    FROM   first_bad AS fb
    WHERE  la.label_trace_id = fb.label_trace_id
       AND la.step_value    > fb.fail_at;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("[post] commit_to_first_anchor_only() finished – monotone fits enforced.")


# def propagate_anchors_topdown(conn):
#     """Implements Method 2 (top‑down propagation) *purely in SQL*.
#
#     The idea is:
#     * start from each trace's smallest *step_value* that still *fits*
#     * let those anchors be the canonical representatives
#     * delete any anchor at a **larger** step that has no earlier TRUE entry inside the
#       same trace.
#
#     After this call, for any given *label_trace_id* the set of rows with `fits=TRUE`
#     forms a chain S < T < …; no orphan TRUEs are left at coarser steps.
#     """
#     sql_drop_tmp = "DROP TABLE IF EXISTS tmp_first_good;"
#     sql_first_good = """
#         CREATE TEMP TABLE tmp_first_good AS
#         SELECT   label_trace_id,
#                  MIN(step_value) AS first_ok_step
#         FROM     label_anchors
#         WHERE    fits = TRUE
#         GROUP BY label_trace_id;
#     """
#     sql_delete_orphans = """
#         DELETE FROM label_anchors AS la
#         USING  tmp_first_good AS fg
#         WHERE  la.label_trace_id = fg.label_trace_id
#           AND  la.step_value   > fg.first_ok_step
#           AND  la.fits = TRUE   -- only delete the stray TRUEs
#           AND NOT EXISTS (
#                 SELECT 1
#                 FROM   label_anchors AS earlier
#                 WHERE  earlier.label_trace_id = la.label_trace_id
#                   AND  earlier.step_value   < la.step_value
#                   AND  earlier.fits = TRUE);
#     """
#     with conn.cursor() as cur:
#         cur.execute(sql_drop_tmp)
#         cur.execute(sql_first_good)
#         cur.execute(sql_delete_orphans)
#     conn.commit()
#     print("[post] propagate_anchors_topdown() finished – subset relation enforced.")


##############################################################################
# Main pipeline
##############################################################################
# def main(do_simplify=False, simplify_tolerance=1.0, font_size=16):
def main(do_simplify=False, simplify_tolerance=1.0, font_size=16,
             enforce_first_fail=True):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.autocommit = True

    # Initialize ScaleStep
    base_scale = 10000
    dataset_name = 'newyan'
    scale_step = ScaleStep(base_scale, dataset_name)

    # 1) Create label_anchors table
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
            angle      DOUBLE PRECISION,
            fits       BOOLEAN  -- Added column
        );
        """)

    # 2) Get faces
    faces = get_faces_of_interest(conn)
    print(f"Found {len(faces)} faces of interest.")

    font_path = "C:/Users/17731/Downloads/Roboto/static/Roboto_Condensed-Light.ttf"
    font = ImageFont.truetype(font_path, font_size)

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
                # roads/water => skeleton approach
                anchors = compute_skeleton_anchors(
                    polygon=poly_shp,
                    do_simplify=do_simplify,
                    simplify_tolerance=simplify_tolerance
                )
                if not anchors:
                    c = poly_shp.centroid
                    anchors = [(c, 0.0)]
            elif BULD_MIN <= fclass < BULD_MAX:
                # building => polylabel approach
                a_pt, a_angle = compute_building_anchor(
                    polygon=poly_shp
                )
                if a_pt is None:
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
                label_text = face_name or ""
                if not label_text.strip():
                    fits = False
                else:
                    # Calculate scale and resolution
                    scale_denominator = scale_step.scale_for_step(S)
                    resolution_mpp = ScaleStep.resolution_mpp(scale_denominator, ppi=96)

                    # Calculate label dimensions in meters
                    bbox = font.getbbox(label_text)
                    width_px = bbox[2] - bbox[0]
                    height_px = bbox[3] - bbox[1]
                    label_width_m = width_px * resolution_mpp
                    label_height_m = height_px * resolution_mpp

                    # Create and rotate rectangle
                    half_w = label_width_m / 2
                    half_h = label_height_m / 2
                    x, y = anchor_pt.x, anchor_pt.y
                    rect = Polygon([
                        (x - half_w, y - half_h),
                        (x + half_w, y - half_h),
                        (x + half_w, y + half_h),
                        (x - half_w, y + half_h)
                    ])
                    rotated_rect = rotate(rect, angle, origin=anchor_pt)

                    if poly_shp and rotated_rect.is_valid and rotated_rect.area > 0:
                        intersection_area = rotated_rect.intersection(poly_shp).area
                        overlap_ratio = intersection_area / rotated_rect.area
                        fits = overlap_ratio >= 0.70  # threshold
                    else:
                        fits = False

                # Insert into database
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO label_anchors(face_id, step_value, feature_class, name, anchor_geom, face_geom, angle, fits)
                        VALUES (%s, %s, %s, %s, ST_GeomFromText(%s, 28992), ST_Multi(ST_GeomFromText(%s, 28992)), %s, %s)
                    """, (face_id, S, fclass, face_name, wkt, poly_wkt, angle, fits))

    # with conn.cursor() as cur:
    #     cur.execute("""
    #         WITH first_failure AS (
    #             SELECT face_id, name, MIN(step_value) AS first_fail_step
    #             FROM label_anchors
    #             WHERE fits = False
    #             GROUP BY face_id, name
    #         )
    #         UPDATE label_anchors la
    #         SET fits = False
    #         FROM first_failure ff
    #         WHERE la.face_id = ff.face_id
    #         AND la.name = ff.name
    #         AND la.step_value >= ff.first_fail_step;
    #     """)

    print("Done! Label anchors inserted.")

    # Done inserting; now assign label_trace_id
    assign_label_trace_ids(conn, distance_threshold=50.0)
    print("Done! label_trace_id assigned.")

    if enforce_first_fail:
        commit_to_first_anchor_only(conn)

    # if propagate_topdown:
    #     propagate_anchors_topdown(conn)

    # Compute 3D bounding boxes
    bounding_boxes = compute_3d_bounding_boxes(conn, create_table=True)
    print("Done! 3D bounding boxes computed.")

    # Visualize them in 3D
    visualize_3d_bounding_boxes(bounding_boxes)

    conn.close()



if __name__ == "__main__":
    # Example usage:
    #   1) with no simplification
    #       main(do_simplify=False)
    #   2) with simplification
    #       main(do_simplify=True, simplify_tolerance=5.0)

    main(do_simplify=True, simplify_tolerance=1.0, font_size=16,
         enforce_first_fail=False)

