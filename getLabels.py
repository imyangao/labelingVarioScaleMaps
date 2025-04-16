import psycopg2
import math
import bisect

##############################################################################
# 1) Database queries to load bounding boxes and anchors
##############################################################################

def load_trace_bounding_boxes(conn):
    """
    Loads 3D bounding boxes in (x, y, step) for each label_trace_id.
    """
    sql = """
        SELECT label_trace_id,
               min_x, max_x,
               min_y, max_y,
               min_step, max_step
          FROM label_trace_3d_bounds
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    # Store in a dict: trace_id -> bounding box
    # bounding box is (min_x, max_x, min_y, max_y, min_step, max_step)
    bb_dict = {}
    for (t_id, mnx, mxx, mny, mxy, mns, mxs) in rows:
        bb_dict[t_id] = (mnx, mxx, mny, mxy, mns, mxs)

    return bb_dict


def load_all_anchors(conn):
    """
    Loads all label anchors, grouped and sorted by step_value.

    Returns a dict:
      trace_id -> list of (step_value, x, y, fits, name)
    """
    sql = """
        SELECT label_trace_id,
               step_value,
               ST_X(anchor_geom) AS x,
               ST_Y(anchor_geom) AS y,
               fits,
               name
          FROM label_anchors
         WHERE label_trace_id IS NOT NULL
         ORDER BY label_trace_id, step_value
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    anchors_by_trace = {}
    for (t_id, step, x, y, f, nm) in rows:
        if t_id not in anchors_by_trace:
            anchors_by_trace[t_id] = []
        anchors_by_trace[t_id].append((step, x, y, f, nm))

    # Ensure sorted by step_value (should already be sorted by the ORDER BY):
    for t_id in anchors_by_trace:
        anchors_by_trace[t_id].sort(key=lambda row: row[0])

    return anchors_by_trace


##############################################################################
# 2) Filtering bounding boxes by user scale
##############################################################################

def get_candidate_traces_for_scale(bb_dict, user_scale):
    """
    Given a dictionary of bounding boxes keyed by trace_id and a user_scale
    (which may be fractional), returns a list of trace_ids that have
    min_step <= user_scale <= max_step (i.e. the scale dimension overlaps).

    We ignore x,y bounding for now; you could also cull by x,y if you know
    the user is centered at (cx, cy) with a certain viewport bounding box.
    """
    candidate_traces = []
    for t_id, (mnx, mxx, mny, mxy, mns, mxs) in bb_dict.items():
        if (user_scale >= mns) and (user_scale <= mxs):
            candidate_traces.append(t_id)

    return candidate_traces


##############################################################################
# 3) Binary search for floor/ceil anchors and interpolation
##############################################################################

def get_interpolated_anchor(anchors_sorted, user_scale):
    """
    anchors_sorted is a list of tuples:
      (step_value, x, y, fits, name)
    sorted by step_value in ascending order.

    We do a binary search to find the anchor(s) that bracket user_scale,
    then interpolate x,y if user_scale is between them.

    Returns (x_interp, y_interp, scale_used, name_used, fits_combined).
    or None if no anchors exist.
    """
    if not anchors_sorted:
        return None

    # Extract just step_value in an array for bisect
    steps = [row[0] for row in anchors_sorted]

    # insertion point
    idx = bisect.bisect_left(steps, user_scale)

    # If user_scale is before the first anchor, clamp:
    if idx == 0:
        (s0, x0, y0, f0, nm0) = anchors_sorted[0]
        return (x0, y0, s0, nm0, f0)

    # If user_scale is beyond the last anchor, clamp:
    if idx >= len(anchors_sorted):
        (sN, xN, yN, fN, nmN) = anchors_sorted[-1]
        return (xN, yN, sN, nmN, fN)

    # Otherwise, user_scale is between anchors_sorted[idx-1] and anchors_sorted[idx]
    (s_floor, x_floor, y_floor, f_floor, nm_floor) = anchors_sorted[idx - 1]
    (s_ceil, x_ceil, y_ceil, f_ceil, nm_ceil)      = anchors_sorted[idx]

    if abs(s_ceil - s_floor) < 1e-9:
        # same step, no interpolation
        return (x_floor, y_floor, s_floor, nm_floor, f_floor)

    # Linear interpolation factor
    t = (user_scale - s_floor) / float(s_ceil - s_floor)

    x_interp = x_floor + t * (x_ceil - x_floor)
    y_interp = y_floor + t * (y_ceil - y_floor)
    fits_combined = f_floor and f_ceil
    # If the name might differ, pick whichever you prefer. We'll pick floor’s:
    name_used = nm_floor

    return (x_interp, y_interp, user_scale, name_used, fits_combined)


##############################################################################
# 4) Transform from world coords to screen coords
#    This is a simple approach to show the logic. If you already use transform.js
#    in the browser, you can skip this and do it client-side.
##############################################################################

def pixel_to_meter(px):
    """
    Approximate: 1 inch = 0.0254 m, 1 inch = 96 px => 1 px ~ 0.00026458 m
    """
    inch = 0.0254
    ppi = 96.0
    return px * (inch / ppi)

def world_to_screen(x_world, y_world,
                    viewport_width, viewport_height,
                    center_world, user_scale):
    """
    Minimal approach:
      - user_scale ~ an integer (or float) that indicates the "zoom level"
      - center_world = (cx, cy) is the map coordinate of the viewport center
      - viewport_width, viewport_height in pixels

    We'll interpret user_scale to mean how many "map units" (like meters in your SRID)
    are shown per meter on screen. That is, if user_scale=12, then 1 meter on screen
    corresponds to 12 map units.

    Then the viewport in "map units" horizontally is:
       pixel_to_meter(viewport_width) * user_scale
    """
    # viewport size in map coords:
    viewport_meters_x = pixel_to_meter(viewport_width) * user_scale
    viewport_meters_y = pixel_to_meter(viewport_height) * user_scale

    # center in world coords:
    cx, cy = center_world

    # scale factors:
    sx = viewport_width / viewport_meters_x
    sy = viewport_height / viewport_meters_y

    # translate relative to center, then scale
    # invert Y so that smaller screen_y is at the top
    screen_x = (x_world - cx) * sx + (viewport_width / 2.0)
    screen_y = (cy - y_world) * sy + (viewport_height / 2.0)

    return (screen_x, screen_y)


##############################################################################
# 5) Putting it all together
##############################################################################

def get_labels_for_zoom(conn, user_scale, viewport_size, center_world):
    """
    High-level function:
      1. Load bounding boxes => quickly find candidate traces that overlap user_scale
      2. For each candidate trace, binary-search for anchor floor/ceil => interpolate
      3. Transform anchor to screen coords
      4. Filter out off-screen or fits=False
    Returns list of (screen_x, screen_y, name, label_trace_id).
    """
    # 1) Load bounding boxes and anchors in memory
    bounding_boxes = load_trace_bounding_boxes(conn)
    anchors_by_trace = load_all_anchors(conn)

    # 1) Check how many bounding boxes we loaded
    print(f"Loaded bounding boxes for {len(bounding_boxes)} label_trace_ids.")

    # 2) Find candidate traces for this user_scale
    candidate_trace_ids = get_candidate_traces_for_scale(bounding_boxes, user_scale)
    print(f"User scale={user_scale}. Candidate traces={len(candidate_trace_ids)}")

    # 3) For each trace, get interpolated anchor
    results = []

    for t_id in candidate_trace_ids:
        anchors_sorted = anchors_by_trace.get(t_id)
        if not anchors_sorted:
            # no anchors for this trace
            # Debug: print something
            print(f"Trace {t_id} is in bounding box range but has no anchors in anchors_by_trace.")
            continue

        anchor_info = get_interpolated_anchor(anchors_sorted, user_scale)
        if not anchor_info:
            print(f"No anchor found (somehow) for trace {t_id}")
            continue

        (wx, wy, step_used, name, fits) = anchor_info

        # Debug check
        if not fits:
            print(f"Trace {t_id}: anchor fits=False => skipping")
            continue

        # world->screen
        (viewport_width, viewport_height) = viewport_size
        (sx, sy) = world_to_screen(wx, wy, viewport_width, viewport_height, center_world, user_scale)

        # Check if on-screen
        on_screen = (0 <= sx <= viewport_width) and (0 <= sy <= viewport_height)
        if not on_screen:
            # Debug
            print(f"Trace {t_id}: anchor is off-screen => skipping. (sx={sx}, sy={sy})")
            continue

        # If we get here, the label is valid
        results.append((sx, sy, name, t_id))

    print(f"Final label count: {len(results)}")
    return results


##############################################################################
# Demo / usage example
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

    # Suppose the user is at scale=12.7, viewport = 800x600 px,
    # centered at some map coords (cx, cy) = (120000, 487000).
    user_scale = 12.7
    viewport_size = (800, 600)
    center_world = (120000.0, 487000.0)

    # Get all the labels that appear at this scale & screen
    label_positions = get_labels_for_zoom(conn, user_scale, viewport_size, center_world)

    for (sx, sy, name, trace_id) in label_positions:
        print(f"Label '{name}' from trace {trace_id} => screen=({sx:.1f},{sy:.1f})")

    conn.close()


if __name__ == "__main__":
    main()
