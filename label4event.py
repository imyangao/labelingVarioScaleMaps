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
        FROM yan_tgap_face f
        LEFT JOIN yan_face y
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
    FROM yan_tgap_edge
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
# Main pipeline
##############################################################################
def main(do_simplify=False, simplify_tolerance=1.0, font_size=16):
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
    dataset_name = 'yan'
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
            angle      DOUBLE PRECISION,
            fits       BOOLEAN  -- Added column
        );
        """)

    # 2) Get faces
    faces = get_faces_of_interest(conn)
    print(f"Found {len(faces)} faces of interest.")

    font_path = "C:/Users/17731/Downloads/Roboto/static/Roboto_Condensed-Light.ttf"
    font = ImageFont.truetype(font_path, font_size)

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
                    # num_chars = len(label_text)
                    # label_width_px = num_chars * 10  # Approx. 10px per character
                    # label_height_px = font_size
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

                    # Check containment
                    fits = rotated_rect.within(poly_shp) if poly_shp else False

                # Insert into database
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO label_anchors(face_id, step_value, feature_class, name, anchor_geom, angle, fits)
                        VALUES (%s, %s, %s, %s, ST_GeomFromText(%s, 28992), %s, %s)
                    """, (face_id, S, fclass, face_name, wkt, angle, fits))

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

    conn.close()
    print("Done! Label anchors inserted.")


if __name__ == "__main__":
    # Example usage:
    #   1) with no simplification
    #       main(do_simplify=False)
    #   2) with simplification
    #       main(do_simplify=True, simplify_tolerance=5.0)

    main(do_simplify=True, simplify_tolerance=10.0, font_size=10)

