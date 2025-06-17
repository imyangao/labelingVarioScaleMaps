import json
import psycopg2
from psycopg2.extras import RealDictCursor   # returns each row as a Python dict


conn = psycopg2.connect(
    dbname="",
    user="",
    password="",
    host="localhost",
    port=5432
)

# pull every record, turning geometries into GeoJSON text
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
    FROM        label_anchors  la
    JOIN        yan_tgap_face  f  ON f.face_id = la.face_id
    WHERE la.name is not null
    ORDER BY    la.label_id;
"""

with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute(sql)
    rows = cur.fetchall()

# write rows to a JSON file
# Each element in rows is already a dict {col: value, …}
out_file = "label_anchors_event.json"
with open(out_file, "w", encoding="utf‑8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(rows):,d} records to {out_file}")
