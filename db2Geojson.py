import psycopg2
import shapely.wkb
import json
from shapely.geometry import mapping

# Database connection info
conn = psycopg2.connect(
    dbname="tgap_test",
    user="postgres",
    password="Gy@001130",
    host="localhost",
    port=5432
)
conn.autocommit = False

# --- Export labels.geojson ---
label_sql = """
    SELECT face_id, step_value, name, angle, fits, ST_AsBinary(anchor_geom)
    FROM label_anchors;
"""

label_features = []
with conn.cursor() as cur:
    cur.execute(label_sql)
    for face_id, step, name, angle, fits, geom_wkb in cur.fetchall():
        pt = shapely.wkb.loads(geom_wkb.tobytes())
        label_features.append({
            "type": "Feature",
            "geometry": mapping(pt),
            "properties": {
                "face_id": face_id,
                "step_value": step,
                "name": name,
                "angle": angle,
                "fits": fits
            }
        })

with open("labels.geojson", "w") as f:
    json.dump({"type": "FeatureCollection", "features": label_features}, f)

print(f"Exported {len(label_features)} labels to labels.geojson")

# --- Export only faces that are referenced in label_anchors, polygonized in PostGIS ---
face_sql = """
    WITH face_edges AS (
    SELECT DISTINCT la.face_id, la.step_value, e.geometry AS geom
    FROM label_anchors la
    JOIN yan_tgap_edge e
        ON (e.left_face_id_low = la.face_id OR e.right_face_id_low = la.face_id
            OR e.left_face_id_high = la.face_id OR e.right_face_id_high = la.face_id)
    JOIN yan_tgap_face f ON f.face_id = la.face_id
    WHERE e.step_high > f.step_low AND e.step_low < f.step_high
    ),
    grouped AS (
        SELECT face_id, step_value, ST_Union(geom) AS merged_geom
        FROM face_edges
        GROUP BY face_id, step_value
    ),
    polygonized AS (
        SELECT face_id, step_value, ST_Polygonize(merged_geom) AS poly_collection
        FROM grouped
        GROUP BY face_id, step_value, merged_geom
    ),
    final AS (
        SELECT p.face_id, p.step_value, (ST_Dump(ST_CollectionExtract(p.poly_collection, 3))).geom AS geom
        FROM polygonized p
    )
    SELECT face_id, step_value, ST_AsBinary(geom) AS geom_wkb
    FROM final;
"""
#
# features = []
# with conn.cursor() as cur:
#     cur.execute(face_sql)
#     for face_id, step_value, geom_wkb in cur.fetchall():
#         poly = shapely.wkb.loads(geom_wkb.tobytes())
#         features.append({
#             "type": "Feature",
#             "geometry": mapping(poly),
#             "properties": {
#                 "face_id": face_id,
#                 "step_value": step_value
#             }
#         })
#
# with open("faces.geojson", "w") as f:
#     json.dump({"type": "FeatureCollection", "features": features}, f)

with conn.cursor(name='faces_cursor') as cur:  # server-side cursor
    cur.execute(face_sql)
    with open("faces.geojson", "w") as f:
        f.write('{"type":"FeatureCollection","features":[')
        first = True
        for face_id, step_value, geom_wkb in cur:
            if not first:
                f.write(',\n')
            first = False
            poly = shapely.wkb.loads(geom_wkb.tobytes())
            json.dump({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "face_id": face_id,
                    "step_value": step_value
                }
            }, f)
        f.write(']}')


# print(f"Exported {len(features)} faces to faces.geojson")



conn.close()