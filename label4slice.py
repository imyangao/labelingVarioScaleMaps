import psycopg2
import geopandas as gpd
import fiona
import os

# Import skeleton-generation function
from genSkeleton import generate_skeleton_for_gpkg


def create_slice_table(conn, step_value=11500):
    """
    Creates a slice table for the specified step_value by running a CREATE TABLE ... AS ...
    statement in PostGIS.
    """
    create_sql = f"""
    DROP TABLE IF EXISTS yan_topo2geom_{step_value}_enriched;
    CREATE TABLE yan_topo2geom_{step_value}_enriched AS
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
        ff.name  -- new
    FROM polygonized_edges p
    JOIN yan_tgap_face f
      ON ST_Contains(p.polygon_geom, f.pip_geometry)
    LEFT JOIN yan_face ff            -- or use JOIN if guaranteed 1:1
      ON ff.face_id = f.face_id      -- new
    WHERE f.step_low <= {step_value} AND f.step_high > {step_value};
    """

    with conn.cursor() as cur:
        cur.execute(create_sql)
    conn.commit()


def export_slice_to_gpkg(conn, step_value=11500, out_gpkg="slice_intermediate.gpkg"):
    """
    Reads the slice table from PostGIS into a GeoDataFrame and writes it to a .gpkg file.
    """
    slice_table_name = f"yan_topo2geom_{step_value}_enriched"
    # sql = f"SELECT face_id, feature_class, polygon_geom FROM {slice_table_name};"
    sql = f"SELECT face_id, feature_class, name, polygon_geom FROM {slice_table_name};"

    # Read PostGIS table as a GeoDataFrame
    gdf = gpd.read_postgis(sql, conn, geom_col="polygon_geom")

    # Write to GeoPackage (intermediate result)
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)

    gdf.to_file(out_gpkg, layer="map_slice", driver="GPKG")
    print(f"Exported slice to {out_gpkg}, layer=map_slice")


def create_label_table_for_step(conn, step_value=11500):
    """
    Creates (if not exists) a label_points_{step_value} table with a label_id SERIAL PK,
    plus step_value, face_id, feature_class, geometry(POINT) for anchors, and angle.
    """
    table_name = f"label_points_{step_value}"
    drop_sql = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
    create_sql = f"""
            CREATE TABLE {table_name} (
                label_id SERIAL PRIMARY KEY,
                step_value INTEGER,
                face_id INTEGER,
                feature_class INTEGER,
                name TEXT,
                anchor_geom geometry(POINT, 28992),
                angle DOUBLE PRECISION
            );
        """

    with conn.cursor() as cur:
        cur.execute(drop_sql)  # drop the old table if it exists
        cur.execute(create_sql)  # create the new one
    conn.commit()


def store_labels_from_gpkg(conn, step_value, gpkg_file):
    """
    Reads label layers from skeleton output .gpkg (e.g. ..._roads_labels, ..._water_labels, etc.)
    and inserts them into label_points_{step_value} with a label_id SERIAL, plus step_value.
    """
    table_name = f"label_points_{step_value}"

    # Check which layers exist in the GPKG
    layers = fiona.listlayers(gpkg_file)

    # Possible label layers your genSkeleton.py might create:
    candidate_label_layers = [
        # example layer names used in genSkeleton.py:
        "map_slice_roads_labels",
        "map_slice_water_labels",
        "map_slice_buildings_centers",
    ]

    # Prepare an INSERT statement
    # We'll just do a straightforward INSERT (we allow multiple anchors for the same face).
    insert_sql = f"""
    INSERT INTO {table_name} 
    (step_value, face_id, feature_class, name, anchor_geom, angle)
    VALUES (%s, %s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 28992), %s)
    """

    cur = conn.cursor()

    for lyr in candidate_label_layers:
        if lyr in layers:
            # Load the label layer as a GeoDataFrame
            gdf = gpd.read_file(gpkg_file, layer=lyr)

            # If your script uses a different attribute name than "angle", adapt accordingly:
            angle_col = "angle" if "angle" in gdf.columns else "rotation"

            for idx, row in gdf.iterrows():
                face_id = row.get("face_id")
                fclass = row.get("feature_class")
                name = row.get("name")  # new
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
    cur.close()


def main():
    conn = psycopg2.connect(
        dbname="tgap_test",
        user="postgres",
        password="Gy@001130",
        host="localhost",
        port=5432
    )
    conn.autocommit = True

    # 1) Set step value
    step_value = 3666

    # 2) Create the slice table in the database
    create_slice_table(conn, step_value=step_value)

    # 3) Export that slice to a GeoPackage (intermediate result)
    intermediate_gpkg = f"slice_intermediate_{step_value}.gpkg"
    export_slice_to_gpkg(conn, step_value=step_value, out_gpkg=intermediate_gpkg)

    # 4) Generate the skeleton-labeled output
    output_gpkg = f"skeleton_output_{step_value}.gpkg"
    if os.path.exists(output_gpkg):
        os.remove(output_gpkg)

    generate_skeleton_for_gpkg(
        input_gpkg=intermediate_gpkg,
        output_gpkg=output_gpkg,
        do_simplify=True,
        simplify_tolerance=10.0
    )
    print(f"Skeleton generation complete. Results in {output_gpkg}.")

    # 5) Create the label table (with label_id serial, step_value, etc.)
    create_label_table_for_step(conn, step_value=step_value)

    # 6) Read label layers from the output .gpkg and store them in the new table
    store_labels_from_gpkg(conn, step_value, output_gpkg)

    print(f"Labels inserted into table label_points_{step_value}.")

    # 7) Close connection
    conn.close()


if __name__ == "__main__":
    main()