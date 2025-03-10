import geopandas as gpd
import numpy as np
from sklearn.decomposition import PCA
from shapely.geometry import Polygon, MultiPolygon, LineString, Point


def compute_orientation(geom):
    """Calculate polygon orientation with convex hull simplification"""
    # Simplify using convex hull to reduce noise
    try:
        geom = geom.convex_hull
    except:  # Handle edge cases (e.g., points/lines)
        return np.nan

    if geom.is_empty or geom.geom_type not in ['Polygon', 'MultiPolygon']:
        return np.nan

    # Extract coordinates from simplified geometry
    coords = []
    if geom.geom_type == 'Polygon':
        coords.extend(geom.exterior.coords)
        for interior in geom.interiors:
            coords.extend(interior.coords)
    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:
            coords.extend(part.exterior.coords)
            for interior in part.interiors:
                coords.extend(interior.coords)

    if len(coords) < 2:
        return np.nan

    # Remove duplicates
    coords_array = np.unique(np.array(coords), axis=0)

    # PCA calculation
    centered = coords_array - np.mean(coords_array, axis=0)
    pca = PCA(n_components=2).fit(centered)
    first_pc = pca.components_[0]

    # Angle adjustment to prevent upside-down orientation
    angle_rad = np.arctan2(first_pc[1], first_pc[0])
    angle_deg = np.degrees(angle_rad) % 360  # 0-360 range

    if 180 < angle_deg < 270:
        angle_deg += 180

    return angle_deg % 180  # Final 0-180 range


def compute_orientation_line(geom, angle_deg, line_scale=0.3):
    """Create orientation line with error handling"""
    if np.isnan(angle_deg) or geom.is_empty:
        return None

    try:
        centroid = geom.centroid
        if centroid.is_empty:
            return None

        # Calculate line length based on polygon size
        minx, miny, maxx, maxy = geom.bounds
        diag_length = np.hypot(maxx - minx, maxy - miny)
        line_length = diag_length * line_scale

        # Calculate endpoints
        angle_rad = np.radians(angle_deg)
        dx = 0.5 * line_length * np.cos(angle_rad)
        dy = 0.5 * line_length * np.sin(angle_rad)

        return LineString([
            Point(centroid.x - dx, centroid.y - dy),
            Point(centroid.x + dx, centroid.y + dy)
        ])
    except:
        return None


# Load and process data
gdf = gpd.read_file(r'C:\topnl_test\yan_topo2geom_2500_enriched.gpkg')
gdf['orientation'] = gdf.geometry.apply(compute_orientation)
gdf['centroid_x'] = gdf.geometry.centroid.x
gdf['centroid_y'] = gdf.geometry.centroid.y

# Create orientation lines
gdf['orientation_line'] = gdf.apply(
    lambda row: compute_orientation_line(row.geometry, row.orientation),
    axis=1
)

# Save to GPKG with separate layers
# Layer 1: Polygons with attributes
gdf_poly = gdf[['face_id', 'feature_class', 'orientation', 'centroid_x', 'centroid_y', 'geometry']]
gdf_poly.to_file('pca_output.gpkg', driver='GPKG', layer='polygons')

# Layer 2: Orientation lines
gdf_lines = gdf[['face_id', 'orientation', 'orientation_line']].dropna(subset=['orientation_line'])
gdf_lines = gdf_lines.set_geometry('orientation_line')
gdf_lines.to_file('pca_output.gpkg', driver='GPKG', layer='orientation_lines')