import psycopg2
import geopandas as gpd
from shapely import wkb
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
        f.feature_class
    FROM polygonized_edges p
    JOIN yan_tgap_face f
      ON ST_Contains(p.polygon_geom, f.pip_geometry)
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
    sql = f"SELECT face_id, feature_class, polygon_geom FROM {slice_table_name};"

    # Read PostGIS table as a GeoDataFrame
    gdf = gpd.read_postgis(sql, conn, geom_col="polygon_geom")

    # Write to GeoPackage (intermediate result)
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)  # remove if you want a fresh file each time

    gdf.to_file(out_gpkg, layer="map_slice", driver="GPKG")
    print(f"Exported slice to {out_gpkg}, layer=map_slice")


def main():
    # 1) Connect to PostGIS
    conn = psycopg2.connect(
        dbname="tgap_test",
        user="postgres",
        password="Gy@001130",
        host="localhost",
        port=5432
    )
    conn.autocommit = True  # optional, you can manage transactions however you prefer

    # 2) Create the slice table in the database
    step_value = 3666
    create_slice_table(conn, step_value=step_value)

    # 3) Export that slice to a GeoPackage as intermediate result
    intermediate_gpkg = f"slice_intermediate_{step_value}.gpkg"
    output_gpkg = f"skeleton_output_{step_value}.gpkg"

    export_slice_to_gpkg(conn, step_value=step_value, out_gpkg=intermediate_gpkg)

    # 4) Call the skeleton-generation function from genSkeleton.py
    if os.path.exists(output_gpkg):
        os.remove(output_gpkg)  # ensure a clean file if you want

    # generate_skeleton_for_gpkg expects:
    #   input_gpkg,
    #   output_gpkg,
    #   do_simplify (bool),
    #   simplify_tolerance (float)
    generate_skeleton_for_gpkg(
        input_gpkg=intermediate_gpkg,  # The file we just wrote
        output_gpkg=output_gpkg,
        do_simplify=True,
        simplify_tolerance=10.0
    )

    print(f"Skeleton generation complete. Results in {output_gpkg}.")

    # 5) Close the connection
    conn.close()


if __name__ == "__main__":
    main()