import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import affinity
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from urllib.parse import quote_plus
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
tqdm.pandas()

# ---------- Connect ------------------------------------------------
DB_NAME = "tgap_test"
DB_USER = "postgres"
DB_PASS = "Gy@001130"
DB_HOST = "localhost"
DB_PORT = 5432


# Use quote_plus to properly URL-encode the password
escaped_password = quote_plus(DB_PASS)
DB_URI = f"postgresql://{DB_USER}:{escaped_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URI)

def read_table(name):
    gdf = gpd.read_postgis(
        f"SELECT label_trace_id, step_value, anchor_geom, angle "
        f"FROM {name}",
        con=engine,
        geom_col="anchor_geom",
    )
    gdf["method"] = name           # tag for later split
    return gdf

gdf_event  = read_table("label_anchors")
gdf_slice  = read_table("label_anchors_from_slices")
gdf_all    = pd.concat([gdf_event, gdf_slice], ignore_index=True)

print(f"{len(gdf_event):>6} rows from label_anchors")
print(f"{len(gdf_slice):>6} rows from label_anchors_from_slices")
print("building dense step grid …")


# ---------- Build a dense step grid per trace ----------------------
def interpolate_trace(df_trace: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Given one label_trace_id for one method, return rows for every
    integer step_value between min and max, linearly interpolated.
    """
    df_trace = df_trace.sort_values("step_value")

    # -- 1. Known key steps
    key_steps   = df_trace["step_value"].to_numpy()
    key_coords  = np.vstack(df_trace["anchor_geom"].apply(lambda p: (p.x, p.y)))
    key_angles  = df_trace["angle"].to_numpy()

    # -- 2. Build the *full* step grid (here: all integers)
    full_steps  = np.arange(key_steps.min(), key_steps.max() + 1)

    # -- 3. Interpolate X and Y independently
    xi = np.interp(full_steps, key_steps, key_coords[:, 0])
    yi = np.interp(full_steps, key_steps, key_coords[:, 1])

    # -- 4. Angle interpolation along the *shortest* arc
    # Compute cumulative unwrapped angle series first -----------------
    unwrapped = np.unwrap(np.deg2rad(key_angles))           # radians, unwrap
    interp_ang = np.interp(full_steps, key_steps, unwrapped)
    interp_ang = np.rad2deg(interp_ang) % 360               # back to degrees 0-360

    # -- 5. Build GeoDataFrame result ---------------------------------
    gseries = gpd.GeoSeries(gpd.points_from_xy(xi, yi), crs=df_trace.crs)
    out = gpd.GeoDataFrame({
        "label_trace_id": df_trace["label_trace_id"].iloc[0],
        "method":         df_trace["method"].iloc[0],
        "step_value":     full_steps,
        "anchor_geom":    gseries,
        "angle":          interp_ang,
    }, geometry="anchor_geom", crs=df_trace.crs)

    return out

# interpolated = (
#     gdf_all.groupby(["method", "label_trace_id"], group_keys=False)
#            .apply(interpolate_trace, include_groups=False)
# )
print("interpolating anchors …")
interpolated = (
    gdf_all.groupby(["method", "label_trace_id"], group_keys=False)
           .progress_apply(interpolate_trace, include_groups=True)
)


# ---------- Compute jump size between consecutive steps ------------
def jumps_for_trace(df_trace):
    df_trace = df_trace.sort_values("step_value")
    # Euclidean distance between consecutive anchors
    coords = np.vstack(df_trace["anchor_geom"].apply(lambda p: (p.x, p.y)))
    dist   = np.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(axis=1))
    # Optional: include rotation term, e.g. + 0.5*abs(Δθ)
    # dist += 0.5 * np.abs(np.diff(df_trace["angle"]))
    return pd.DataFrame({
        "label_trace_id": df_trace["label_trace_id"].iloc[0],
        "method":         df_trace["method"].iloc[0],
        "jump":           dist,
    })

# jumps = (
#     interpolated.groupby(["method", "label_trace_id"], group_keys=False)
#                 .apply(jumps_for_trace, include_groups=False)
# )
print("computing jumps …")
jumps = (
    interpolated.groupby(["method", "label_trace_id"], group_keys=False)
                .progress_apply(jumps_for_trace, include_groups=True)
)

big = jumps[(jumps.method=="label_anchors") & (jumps.jump > 1)]   # >1 m
print(big.sort_values("jump", ascending=False).head(20))

threshold = 0.030

# Filter for method 'label_anchors' and jump > 0.030
large_jumps = jumps[(jumps["method"] == "label_anchors") & (jumps["jump"] > threshold)]

# Count them
num_large_jumps = len(large_jumps)
print(f"Number of jumps > {threshold} units in 'label_anchors': {num_large_jumps}")

# ---------- Aggregate statistics -----------------------------------
stats_per_method = (
    jumps.groupby("method")["jump"]
         .agg(mean="mean",
              # median="median",
              p99=lambda s: np.percentile(s, 99),
              p95=lambda s: np.percentile(s, 95),
              p90=lambda s: np.percentile(s, 90),
              maximum="max",
              # std="std",
              n_samples="size")
         .reset_index()
)

print("\n=== Smoothness statistics per method (units of CRS) ===")
print(stats_per_method.to_markdown(index=False, floatfmt=".3f"))

