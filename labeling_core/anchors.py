import math
from shapely.geometry import Point, LineString
from shapely.algorithms.polylabel import polylabel
from .skeleton import (
    build_skeleton_lines,
    lines_to_graph,
    get_junction_nodes,
    find_junction_to_junction_paths,
)

def compute_line_angle(line):
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    (x1, y1), (x2, y2) = coords[0], coords[-1]
    dx, dy = (x2 - x1), (y2 - y1)
    angle_deg = math.degrees(math.atan2(dy, dx))
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
    return angle_deg

def compute_skeleton_anchors(polygon, do_simplify=False, simplify_tolerance=1.0, threshold_fraction=0.5):
    """
    Computes label anchors for a polygon using its skeleton.
    """
    if do_simplify and simplify_tolerance > 0:
        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

    raw_skel_lines = build_skeleton_lines(polygon)
    if not raw_skel_lines:
        return []

    G = lines_to_graph(raw_skel_lines)
    junctions = get_junction_nodes(G, min_degree=3)
    
    candidate_lines = []
    if len(junctions) < 2:
        # Not enough junctions - no very important segment
        return []
    else:
        primary_paths = find_junction_to_junction_paths(G, junctions)
        if not primary_paths:
            return []
        candidate_lines = sorted(primary_paths, key=lambda l: l.length, reverse=True)

    if not candidate_lines:
        return []

    max_len = candidate_lines[0].length
    length_threshold = threshold_fraction * max_len

    out_anchors = []
    for ln in candidate_lines:
        if ln.length < length_threshold:
            break
        midpt = ln.interpolate(0.5, normalized=True)
        angle = compute_line_angle(ln)
        out_anchors.append((midpt, angle))
    return out_anchors


def compute_building_anchor(polygon):
    """
    Computes a single label anchor for a building polygon.
    The anchor point is the centroid (if inside) or polylabel.
    The angle is from the longest side of the minimum rotated rectangle.
    """
    if polygon.is_empty:
        return []

    # Try centroid first
    centroid = polygon.centroid
    if polygon.contains(centroid):
        anchor = centroid
    else:
        # Fallback to polylabel only if centroid is not inside
        anchor = polylabel(polygon, tolerance=1.0)

    minrect = polygon.minimum_rotated_rectangle
    coords = list(minrect.exterior.coords)
    if len(coords) < 2:
        return [(anchor, 0.0)]

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
            
    return [(anchor, best_angle)] 