import json
import psycopg2
from psycopg2.extras import RealDictCursor   # returns each row as a Python dict

# -- 1. database connection ----------------------------------------------------
conn = psycopg2.connect(
    dbname="tgap_test",
    user="postgres",
    password="Gy@001130",
    host="localhost",
    port=5432
)

# -- 2. pull every record, turning geometries into GeoJSON text ----------------
sql = """
    SELECT
        la.label_id,
        la.face_id,
        la.step_value,
        f.step_high,                       
        la.feature_class,
        la.name,
        ST_AsGeoJSON(la.anchor_geom)::json AS anchor_geom,
        la.angle,
        la.label_trace_id
    FROM        label_anchors_from_slices  la
    JOIN        yan_tgap_face  f  ON f.face_id = la.face_id
    WHERE la.name is not null
    ORDER BY    la.label_id;
"""

with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute(sql)
    rows = cur.fetchall()

# -- 3. write rows to a prettified JSON file -----------------------------------
#     Each element in rows is already a dict {col: value, …}
out_file = "label_anchors.json"
with open(out_file, "w", encoding="utf‑8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(rows):,d} records to {out_file}")
