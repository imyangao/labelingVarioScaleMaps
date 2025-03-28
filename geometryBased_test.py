import psycopg2

##############################################################################
# Database connection info
##############################################################################
DB_NAME = "your_database"
DB_USER = "postgres"
DB_PASS = "your_password"
DB_HOST = "localhost"
DB_PORT = 5432

##############################################################################
# Feature-class ranges
##############################################################################
ROAD_MIN, ROAD_MAX = 10000, 11000
WATER_MIN, WATER_MAX = 12000, 13000
BULD_MIN, BULD_MAX = 13000, 14000


##############################################################################
# 1) Retrieve all faces of interest
##############################################################################
def get_faces_of_interest(conn):
    sql = f"""
    SELECT face_id, step_low, step_high, feature_class
    FROM yan_tgap_face
    WHERE (feature_class >= {ROAD_MIN} AND feature_class < {ROAD_MAX})
       OR (feature_class >= {WATER_MIN} AND feature_class < {WATER_MAX})
       OR (feature_class >= {BULD_MIN}  AND feature_class < {BULD_MAX})
    ORDER BY face_id;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return rows


##############################################################################
# 2) Retrieve edges that *might* bound a given face
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
      step_high > %s   -- edge ends after face starts
      AND step_low < %s -- edge starts before face ends
      AND (
           left_face_id_low = %s
        OR right_face_id_low = %s
        OR left_face_id_high = %s  -- optional, if you want to catch future changes
        OR right_face_id_high = %s -- optional
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
# 3) Figure out sub-intervals in which face_id actually bounds an edge
##############################################################################
def intervals_where_edge_bounds_face_noSwitch(
        face_id,
        face_step_low,
        face_step_high,
        edge_record
):

    (edge_id,
     e_low, e_high,
     start_node_id, end_node_id,
     lf_low, rf_low,
     lf_high, rf_high,
     edge_class,
     geom_wkb) = edge_record

    # Overlap with face's [face_step_low..face_step_high)
    overall_start = max(e_low, face_step_low)
    overall_end = min(e_high, face_step_high)
    if overall_start >= overall_end:
        return  # no time overlap

    # If face_id matches the left side for [e_low..e_high),
    # then within that time range, it bounds the face.
    if lf_low == face_id:
        yield (overall_start, overall_end, geom_wkb)

    # If face_id matches the right side for [e_low..e_high),
    if rf_low == face_id:
        yield (overall_start, overall_end, geom_wkb)



##############################################################################
# 4) Polygonize lines
##############################################################################
def polygonize_in_postgis_and_get_centroid(conn, line_wkbs):
    if not line_wkbs:
        return None  # no lines => no polygon

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
        SELECT ST_AsBinary(ST_Centroid(poly)) AS anchor_wkb
        FROM polys
        WHERE NOT ST_IsEmpty(poly);
        """
        # We pass line_wkbs as an array to unnest(...)
        cur.execute(sql, (line_wkbs,))
        row = cur.fetchone()
        if not row:
            return None
        anchor_wkb = row[0]  # This is the WKB for the centroid point
        return anchor_wkb


##############################################################################
# 5) Main pipeline
##############################################################################
def main():
    conn = psycopg2.connect(
        dbname="tgap_test",
        user="postgres",
        password="Gy@001130",
        host="localhost",
        port=5432
    )
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS label_anchors (
            label_id   SERIAL PRIMARY KEY,
            face_id    INTEGER,
            step_value INTEGER,
            anchor_geom geometry(POINT, 28992),
            angle      DOUBLE PRECISION
        );
        """)

    # 1) Fetch faces
    faces = get_faces_of_interest(conn)
    print(f"Found {len(faces)} faces of interest.")

    for (face_id, f_low, f_high, fclass) in faces:
        # 2) Edges that might bound this face
        edges = get_edges_for_face(conn, face_id, f_low, f_high)
        if not edges:
            continue

        # 3) Build sub-intervals and event steps
        edge_intervals = []
        event_steps = {f_low, f_high}

        for e in edges:
            for (start, end, wkb_line) in intervals_where_edge_bounds_face_noSwitch(
                    face_id, f_low, f_high, e
            ):
                edge_intervals.append((start, end, wkb_line))
                event_steps.add(start)
                event_steps.add(end)

        # Sort the time boundaries
        sorted_events = sorted(event_steps)

        # 4) Step through each stable interval [S..T)
        for i in range(len(sorted_events) - 1):
            S = sorted_events[i]
            T = sorted_events[i + 1]
            if S == T:
                continue

            step_pick = S
            if step_pick >= f_high:
                # face doesn't exist at or beyond f_high
                continue

            # Collect lines bounding face at step_pick
            boundary_wkbs = []
            for (start, end, line_wkb) in edge_intervals:
                if start <= step_pick < end:
                    boundary_wkbs.append(line_wkb)

            if not boundary_wkbs:
                continue

            # 5) Polygonize in PostGIS & get centroid
            anchor_wkb = polygonize_in_postgis_and_get_centroid(conn, boundary_wkbs)
            if anchor_wkb is None:
                continue  # no polygon formed

            # Store a trivial angle=0.0
            angle = 0.0

            # 6) Insert label
            with conn.cursor() as cur:
                insert_sql = """
                INSERT INTO label_anchors(face_id, step_value, anchor_geom, angle)
                VALUES (%s, %s, ST_GeomFromWKB(%s, 28992), %s)
                """
                cur.execute(insert_sql, (face_id, step_pick, anchor_wkb, angle))

    conn.close()
    print("Done! Label anchors inserted.")


if __name__ == "__main__":
    main()