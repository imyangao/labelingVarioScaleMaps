import geopandas as gpd
import os
import fiona
import networkx as nx
from shapely.geometry import LineString, Point
from math import sqrt, acos, degrees
from tri.delaunay.helpers import ToPointsAndSegments
from grassfire import calc_skel
from math import pi
from shapely.algorithms.polylabel import polylabel
import math


def largest_inscribed_circle(polygon):
    if polygon.geom_type not in ["Polygon", "MultiPolygon"]:
        return None, None

    try:
        # Try centroid first
        centroid = polygon.centroid
        if polygon.contains(centroid):
            center = centroid
            center = centroid
        else:
            # Fallback to polylabel only if centroid is not inside
            center = polylabel(polygon, tolerance=1.0)
        
        radius = polygon.boundary.distance(center)
        return center, radius
    except Exception as e:
        print("Polylabel failed:", e)
        return None, None



# ----------------------------------------------------------------------
# Step A: Build the raw skeleton for a polygon
# ----------------------------------------------------------------------
def extract_boundary_segments(polygon):
    """Extracts boundary segments from a Polygon or MultiPolygon."""
    conv = ToPointsAndSegments()

    if polygon.geom_type == "Polygon":
        boundaries = [polygon.exterior] + list(polygon.interiors)
    elif polygon.geom_type == "MultiPolygon":
        # Collect exteriors + interiors from each sub-polygon
        boundaries = [p.exterior for p in polygon.geoms] + [i for p in polygon.geoms for i in p.interiors]
    else:
        return None

    for boundary in boundaries:
        coords = list(boundary.coords)
        for i in range(len(coords) - 1):
            start, end = coords[i], coords[i + 1]
            conv.add_point(start)
            conv.add_point(end)
            conv.add_segment(start, end)

    return conv

def build_skeleton_lines(polygon):
    """Return a list of skeleton LineStrings for a single polygon using Grassfire."""
    conv = extract_boundary_segments(polygon)
    if conv is None:
        return []
    skel = calc_skel(conv, internal_only=True)

    skeletons = []
    for segment in skel.segments():
        (start, end), _, = segment
        skeletons.append(LineString([start, end]))
    return skeletons

# ----------------------------------------------------------------------
# Step B: Convert skeleton lines to a graph
# ----------------------------------------------------------------------
def lines_to_graph(lines):
    """
    Build a NetworkX graph from a list of LineStrings.
    Consecutive coordinate pairs become edges.
    """
    G = nx.Graph()
    for line in lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            G.add_node(p1)
            G.add_node(p2)
            G.add_edge(p1, p2)
    return G

# ----------------------------------------------------------------------
# Step C: Identify high-degree junctions
# ----------------------------------------------------------------------
def get_junction_nodes(G, min_degree=3):
    """
    Return the set of all nodes in graph G whose degree >= min_degree.
    """
    return {n for n, deg in G.degree() if deg >= min_degree}

# ----------------------------------------------------------------------
# Step D: Find paths connecting the high-degree junctions (removing leaves)
# ----------------------------------------------------------------------
def find_junction_to_junction_paths(G, junctions):
    """
    For every pair of junctions (in the same connected component),
    find a simple path that does not pass through any *other* junction.
    This yields direct connections between major junctions.
    """
    # Map each node to its connected component
    connected_comps = list(nx.connected_components(G))
    node_to_cc = {}
    for i, comp in enumerate(connected_comps):
        for node in comp:
            node_to_cc[node] = i

    junction_list = sorted(list(junctions), key=lambda x: (x[0], x[1]))
    linestrings = []

    for i in range(len(junction_list)):
        for j in range(i+1, len(junction_list)):
            j1 = junction_list[i]
            j2 = junction_list[j]
            # Skip if not in same connected component
            if node_to_cc[j1] != node_to_cc[j2]:
                continue
            # Avoid passing through other junctions
            path = find_path_excluding_junctions(G, j1, j2, junctions)
            if path is not None and len(path) > 1:
                linestrings.append(LineString(path))
    return linestrings

def find_path_excluding_junctions(G, start, goal, junction_set):
    """
    Simple BFS from start to goal, forbidding visits to any *other* junction.
    If we encounter a junction in mid-path (not the goal), skip that route.
    Returns a list of coordinates forming the path, or None if none found.
    """
    from collections import deque
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path
        for neighbor in G.neighbors(current):
            if neighbor in visited:
                continue
            # If neighbor is a junction but not the goal, skip
            if neighbor in junction_set and neighbor != goal:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return None

# ----------------------------------------------------------------------
# Step E: Merge lines that are nearly collinear
# ----------------------------------------------------------------------
def angle_between_vectors(v1, v2):
    """Acute angle between two 2D vectors in radians."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = sqrt(v1[0]**2 + v1[1]**2)
    mag2 = sqrt(v2[0]**2 + v2[1]**2)
    if mag1 * mag2 == 0:
        return 0.0
    cos_val = dot / (mag1 * mag2)
    cos_val = max(min(cos_val, 1.0), -1.0)  # Clamp to avoid numerical issues
    angle_rad = acos(cos_val)
    acute_angle = min(angle_rad, pi - angle_rad)  # Get the acute angle
    return acute_angle

def angle_between_lines(line1, line2):
    """
    Return the absolute acute angle (in degrees) between two lines at their shared node,
    or None if they do not share an endpoint.
    """
    coords1 = list(line1.coords)
    coords2 = list(line2.coords)
    endpoints1 = {coords1[0], coords1[-1]}
    endpoints2 = {coords2[0], coords2[-1]}
    shared = endpoints1.intersection(endpoints2)
    if not shared:
        return None
    shared_point = list(shared)[0]

    def get_other_endpoint(line, sp):
        c = list(line.coords)
        if c[0] == sp:
            return (c[-1][0] - sp[0], c[-1][1] - sp[1])
        else:
            return (c[0][0] - sp[0], c[0][1] - sp[1])

    v1 = get_other_endpoint(line1, shared_point)
    v2 = get_other_endpoint(line2, shared_point)
    rad = angle_between_vectors(v1, v2)
    return degrees(rad)  # Acute angle in degrees

def merge_two_lines(line1, line2):
    """
    Merge line1 and line2 into a single LineString if they share an endpoint
    and are nearly collinear. Return the union as a 2-point line from
    the outermost endpoints.
    """
    c1 = list(line1.coords)
    c2 = list(line2.coords)
    # Combine coords, removing duplicates but preserving order
    # just gather them, then find the two that are farthest apart.
    combined = c1 + c2
    combined = list(dict.fromkeys(combined))  # remove duplicates in insertion order
    if len(combined) < 2:
        return None
    # Find the two points in combined that are farthest apart
    best_dist = 0.0
    best_pair = None
    for i in range(len(combined)):
        for j in range(i+1, len(combined)):
            dist = Point(combined[i]).distance(Point(combined[j]))
            if dist > best_dist:
                best_dist = dist
                best_pair = (combined[i], combined[j])
    if best_pair is None:
        return None
    return LineString([best_pair[0], best_pair[1]])

def merge_collinear_lines(lines, angle_threshold=5.0):
    """
    Iteratively merge lines that share a node and have angle < angle_threshold.
    """
    merged_any = True
    out_lines = list(lines)

    while merged_any:
        merged_any = False
        used = set()
        new_list = []
        i = 0
        while i < len(out_lines):
            if i in used:
                i += 1
                continue
            lineA = out_lines[i]
            merged_line = None

            # Attempt to merge with another line
            for j in range(i+1, len(out_lines)):
                if j in used:
                    continue
                lineB = out_lines[j]
                ang = angle_between_lines(lineA, lineB)
                if ang is not None and ang < angle_threshold:
                    # Merge them
                    mline = merge_two_lines(lineA, lineB)
                    if mline:
                        merged_line = mline
                        used.add(i)
                        used.add(j)
                        merged_any = True
                        break

            if merged_line:
                new_list.append(merged_line)
            else:
                # No merge for lineA
                new_list.append(lineA)
            used.add(i)
            i += 1

        out_lines = new_list

    return out_lines


# ----------------------------------------------------------------------
# Step F: Putting it all together in generate_skeleton_for_gpkg
# ----------------------------------------------------------------------
def generate_skeleton_for_gpkg(
        input_gpkg,
        output_gpkg,
        do_simplify=False,
        simplify_tolerance=0.5
):
    if not os.path.exists(input_gpkg):
        print(f"Error: Input file {input_gpkg} does not exist.")
        return

    layers = fiona.listlayers(input_gpkg)
    print("Available layers:", layers)

    if not layers:
        print("Error: No layers found in the input GPKG.")
        return

    if os.path.exists(output_gpkg):
        os.remove(output_gpkg)

    for layer in layers:
        print(f"Processing layer: {layer}")
        gdf = gpd.read_file(input_gpkg, layer=layer)

        # Keep only valid polygons and explode MultiPolygons
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        gdf = gdf[gdf.geometry.is_valid]
        gdf = gdf.explode(index_parts=True)
        gdf = gdf.reset_index(drop=True)

        if gdf.empty:
            print(f"Skipping layer {layer}, no valid polygons found.")
            continue

        # Prepare output collections
        layer_simplified_polys = []
        layer_roads_skeleton = []
        layer_roads_mainskel = []

        layer_buildings_centers = []

        layer_roads_labels = []

        layer_water_skeleton = []
        layer_water_mainskel = []
        layer_water_labels = []

        for idx, row in gdf.iterrows():
            try:
                polygon = row.geometry.buffer(0)  # Fix geometry
                feature_class = row.get('feature_class', None)

                # Skip if feature_class is missing or invalid
                if feature_class is None:
                    continue

                if 10000 <= feature_class < 11000:
                    # Optional polygon simplification
                    if do_simplify and simplify_tolerance > 0:
                        poly_for_skel = polygon.simplify(simplify_tolerance, preserve_topology=True)
                        layer_simplified_polys.append({"geometry": poly_for_skel, "poly_id": idx})
                    else:
                        poly_for_skel = polygon

                    # Build raw skeleton
                    raw_skel_lines = build_skeleton_lines(poly_for_skel)
                    if not raw_skel_lines:
                        continue

                    # Store the raw skeleton lines if you want them in output
                    for ln in raw_skel_lines:
                        layer_roads_skeleton.append({"geometry": ln, "poly_id": idx})

                    # Convert to graph -> find junctions -> connect them
                    G = lines_to_graph(raw_skel_lines)
                    junctions = get_junction_nodes(G, min_degree=3)
                    if len(junctions) < 2:
                        # not enough skeleton complexity
                        continue

                    primary_paths = find_junction_to_junction_paths(G, junctions)
                    # Using primary_paths directly without merging collinear lines

                    # Store primary skeleton lines
                    for ln in primary_paths:
                        if ln.length > 0:
                            layer_roads_mainskel.append({"geometry": ln, "poly_id": idx})

                    # LABELING: Take all lines above a fraction of the max length
                    sorted_lines = sorted(primary_paths, key=lambda l: l.length, reverse=True)
                    if len(sorted_lines) > 0:
                        max_length = sorted_lines[0].length
                        threshold_fraction = 0.50  # label lines >= 25% of the longest line
                        length_threshold = threshold_fraction * max_length
                    else:
                        length_threshold = 0.0

                    for line in sorted_lines:
                        if line.length < length_threshold:
                            break  # lines are sorted desc, so break as soon as below threshold

                        # Compute midpoint and angle
                        midpoint = line.interpolate(0.5, normalized=True)
                        (x1, y1) = line.coords[0]
                        (x2, y2) = line.coords[-1]
                        dx = x2 - x1
                        dy = y2 - y1
                        angle_deg = math.degrees(math.atan2(dy, dx))
                        # Constrain angle to -90..90
                        if angle_deg > 90:
                            angle_deg -= 180
                        elif angle_deg < -90:
                            angle_deg += 180

                        layer_roads_labels.append({
                            "geometry": midpoint,
                            "angle": angle_deg,
                            "feature_class": row['feature_class'],
                            "face_id": row['face_id'],
                            "name": row['name'],
                            "poly_id": idx
                        })

                elif 12000 <= feature_class < 13000:
                    # Optional polygon simplification
                    if do_simplify and simplify_tolerance > 0:
                        poly_for_skel = polygon.simplify(simplify_tolerance, preserve_topology=True)
                    else:
                        poly_for_skel = polygon

                    # Build raw skeleton
                    raw_skel_lines = build_skeleton_lines(poly_for_skel)
                    if not raw_skel_lines:
                        continue

                    # Store raw skeleton lines if desired
                    for ln in raw_skel_lines:
                        layer_water_skeleton.append({"geometry": ln, "poly_id": idx})

                    # Convert to graph -> find junctions -> connect them
                    G = lines_to_graph(raw_skel_lines)
                    junctions = get_junction_nodes(G, min_degree=3)
                    if len(junctions) < 2:
                        continue

                    primary_paths = find_junction_to_junction_paths(G, junctions)
                    # Using primary_paths directly without merging collinear lines

                    # Store primary skeleton lines
                    for ln in primary_paths:
                        if ln.length > 0:
                            layer_water_mainskel.append({"geometry": ln, "poly_id": idx})

                    # LABELING: multiple lines above length threshold
                    sorted_lines = sorted(primary_paths, key=lambda l: l.length, reverse=True)
                    if len(sorted_lines) > 0:
                        max_length = sorted_lines[0].length
                        threshold_fraction = 0.50
                        length_threshold = threshold_fraction * max_length
                    else:
                        length_threshold = 0.0

                    for line in sorted_lines:
                        if line.length < length_threshold:
                            break

                        midpoint = line.interpolate(0.5, normalized=True)
                        (x1, y1) = line.coords[0]
                        (x2, y2) = line.coords[-1]
                        dx = x2 - x1
                        dy = y2 - y1
                        angle_deg = math.degrees(math.atan2(dy, dx))
                        if angle_deg > 90:
                            angle_deg -= 180
                        elif angle_deg < -90:
                            angle_deg += 180

                        layer_water_labels.append({
                            "geometry": midpoint,
                            "angle": angle_deg,
                            "feature_class": row['feature_class'],
                            "face_id": row['face_id'],
                            "name": row['name'],
                            "poly_id": idx
                        })

                # Process Buildings (13000 <= feature_class < 14000)
                elif 13000 <= feature_class < 14000:
                    center, radius = largest_inscribed_circle(polygon)
                    if center is not None:
                        # Calculate rotation angle from minimum rotated rectangle
                        rect = polygon.minimum_rotated_rectangle
                        coords = list(rect.exterior.coords)
                        max_length = 0.0
                        best_angle = 0.0
                        for i in range(len(coords) - 1):
                            p1 = coords[i]
                            p2 = coords[i + 1]
                            length = Point(p1).distance(Point(p2))
                            if length > max_length:
                                max_length = length
                                dx = p2[0] - p1[0]
                                dy = p2[1] - p1[1]
                                angle_rad = math.atan2(dy, dx)
                                best_angle = math.degrees(angle_rad)
                        # Adjust angle to be within -90 to 90 degrees
                        if best_angle > 90:
                            best_angle -= 180
                        elif best_angle < -90:
                            best_angle += 180

                        layer_buildings_centers.append({
                            "geometry": center,
                            "radius": radius,
                            "angle": best_angle,
                            "feature_class": row['feature_class'],
                            "face_id": row['face_id'],
                            "name": row['name'],
                            "poly_id": idx
                        })

            except Exception as e:
                print(f"Skipping polygon at index {idx} due to error: {e}")

        # Write simplified polygons (roads only)
        if do_simplify and layer_simplified_polys:
            simp_gdf = gpd.GeoDataFrame(layer_simplified_polys, crs=gdf.crs)
            simp_layer_name = f"{layer}_simplified"
            simp_gdf.to_file(output_gpkg, layer=simp_layer_name, driver="GPKG")

        # Write road skeletons
        if layer_roads_skeleton:
            skel_gdf = gpd.GeoDataFrame(layer_roads_skeleton, crs=gdf.crs)
            skel_gdf.to_file(output_gpkg, layer=f"{layer}_roads_skeleton", driver="GPKG")
        if layer_roads_mainskel:
            mainskel_gdf = gpd.GeoDataFrame(layer_roads_mainskel, crs=gdf.crs)
            mainskel_gdf.to_file(output_gpkg, layer=f"{layer}_roads_mainskel", driver="GPKG")

        if layer_roads_labels:
            road_labels_gdf = gpd.GeoDataFrame(layer_roads_labels, crs=gdf.crs)
            road_labels_gdf.to_file(output_gpkg, layer=f"{layer}_roads_labels", driver="GPKG")

        if layer_water_skeleton:
            w_skel_gdf = gpd.GeoDataFrame(layer_water_skeleton, crs=gdf.crs)
            w_skel_gdf.to_file(output_gpkg, layer=f"{layer}_water_skeleton", driver="GPKG")

        if layer_water_mainskel:
            w_mainskel_gdf = gpd.GeoDataFrame(layer_water_mainskel, crs=gdf.crs)
            w_mainskel_gdf.to_file(output_gpkg, layer=f"{layer}_water_mainskel", driver="GPKG")

        if layer_water_labels:
            w_labels_gdf = gpd.GeoDataFrame(layer_water_labels, crs=gdf.crs)
            w_labels_gdf.to_file(output_gpkg, layer=f"{layer}_water_labels", driver="GPKG")

        # Write building centers
        if layer_buildings_centers:
            bld_gdf = gpd.GeoDataFrame(layer_buildings_centers, crs=gdf.crs)
            bld_gdf.to_file(output_gpkg, layer=f"{layer}_buildings_centers", driver="GPKG")

    print("Processing completed.")


# ----------------------------------------------------------------------
# Example usage:
# ----------------------------------------------------------------------
if __name__ == "__main__":
    input_gpkg = r"C:\topnl_test\yan_topo2geom_2500_enriched.gpkg"
    output_gpkg = r"skeleton_2500_test.gpkg"

    generate_skeleton_for_gpkg(
        input_gpkg,
        output_gpkg,
        do_simplify=False,  # enable polygon simplification
        simplify_tolerance=0.0  # set a tolerance
    )