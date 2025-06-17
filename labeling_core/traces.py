import math
from collections import defaultdict
from itertools import groupby
import psycopg2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def assign_label_trace_ids(conn, table_name, distance_per_step=float('inf')):
    """
    Assign label_trace_id to label anchors across steps per face using closest-point matching.
    Allows greater spatial deviation for anchors with larger step gaps.

    Parameters:
    - conn: Active database connection.
    - table_name: The name of the label anchors table to process.
    - distance_per_step: How many map units are allowed per step in time.
    """
    # 1. Add column if not exists
    with conn.cursor() as cur:
        try:
            # Use ASNI SQL standard for adding column if not exists
            cur.execute(f"""
                ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS label_trace_id BIGSERIAL;
            """)
        except psycopg2.errors.DuplicateColumn:
            conn.rollback()
            # Try to add BIGSERIAL if previous command fails
            try:
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN label_trace_id BIGSERIAL;")
            except psycopg2.errors.DuplicateColumn:
                conn.rollback() # Column already exists
        except Exception as e:
            print(f"Could not add label_trace_id column to {table_name}: {e}")
            conn.rollback()

    # 2. Fetch all anchors
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT label_id, face_id, step_value,
                   ST_X(anchor_geom) AS x,
                   ST_Y(anchor_geom) AS y
            FROM {table_name}
            ORDER BY face_id, step_value;
        """)
        rows = cur.fetchall()

    # 3. Group by face_id
    faces_map = defaultdict(list)
    for (lbl_id, face_id, step_val, x, y) in rows:
        faces_map[face_id].append((lbl_id, step_val, x, y))

    assignments = {}
    next_trace_id = 1

    for face_id, anchors in faces_map.items():
        anchors.sort(key=lambda r: r[1])  # sort by step_value
        steps_data = [(step_val, list(grp)) for step_val, grp in groupby(anchors, key=lambda r: r[1])]

        active_traces = []  # (trace_id, x, y)
        prev_step_val = None

        for idx, (step_val, anchor_list) in enumerate(steps_data):
            if idx == 0:
                for (lbl_id, _, x, y) in anchor_list:
                    assignments[lbl_id] = next_trace_id
                    active_traces.append((next_trace_id, x, y))
                    next_trace_id += 1
                prev_step_val = step_val
            else:
                step_gap = step_val - prev_step_val
                max_dist = step_gap * distance_per_step

                new_active = []
                matched_prev = set()

                for (lbl_id, _, x, y) in anchor_list:
                    min_dist = float('inf')
                    best_idx = -1

                    for i, (trace_id, px, py) in enumerate(active_traces):
                        if i in matched_prev:
                            continue
                        dist = math.hypot(x - px, y - py)
                        if dist < min_dist and dist <= max_dist:
                            min_dist = dist
                            best_idx = i

                    if best_idx != -1:
                        trace_id = active_traces[best_idx][0]
                        assignments[lbl_id] = trace_id
                        matched_prev.add(best_idx)
                        new_active.append((trace_id, x, y))
                    else:
                        assignments[lbl_id] = next_trace_id
                        new_active.append((next_trace_id, x, y))
                        next_trace_id += 1

                active_traces = new_active
                prev_step_val = step_val

    # 4. Write results to DB
    with conn.cursor() as cur:
        for lbl_id, trace_id in assignments.items():
            cur.execute(f"""
                UPDATE {table_name}
                   SET label_trace_id = %s
                 WHERE label_id = %s;
            """, (trace_id, lbl_id))
    conn.commit()
    print(f"label_trace_id assigned to {table_name} using closest-point matching.")


def compute_3d_bounding_boxes(conn, anchors_table, bounds_table, create_table=True):
    """
    For each label_trace_id, compute an axis-aligned bounding box in x, y,
    and step_value space.
    """
    if create_table:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {bounds_table};")
            cur.execute(f"""
                CREATE TABLE {bounds_table} (
                    label_trace_id BIGINT,
                    min_x DOUBLE PRECISION,
                    max_x DOUBLE PRECISION,
                    min_y DOUBLE PRECISION,
                    max_y DOUBLE PRECISION,
                    min_step INTEGER,
                    max_step INTEGER
                );
            """)
        conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT label_trace_id,
                   ST_X(anchor_geom) as x,
                   ST_Y(anchor_geom) as y,
                   step_value
            FROM {anchors_table}
            WHERE label_trace_id IS NOT NULL
            ORDER BY label_trace_id;
        """)
        rows = cur.fetchall()

    trace_map = defaultdict(list)
    for t_id, x, y, stp in rows:
        trace_map[t_id].append((x, y, stp))

    bounding_boxes = {}
    for t_id, coords in trace_map.items():
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        steps = [c[2] for c in coords]
        bounding_boxes[t_id] = (min(xs), max(xs), min(ys), max(ys), min(steps), max(steps))

    if create_table:
        with conn.cursor() as cur:
            for t_id, (mnx, mxx, mny, mxy, mns, mxs) in bounding_boxes.items():
                cur.execute(f"""
                    INSERT INTO {bounds_table} (
                        label_trace_id,
                        min_x, max_x, min_y, max_y,
                        min_step, max_step
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, (t_id, mnx, mxx, mny, mxy, mns, mxs))
        conn.commit()

    return bounding_boxes


def visualize_3d_bounding_boxes(bounding_boxes):
    """
    Create a simple 3D plot of each axis-aligned bounding box.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Step')

    for t_id, (min_x, max_x, min_y, max_y, min_step, max_step) in bounding_boxes.items():
        corners = [
            (min_x, min_y, min_step), (min_x, min_y, max_step),
            (min_x, max_y, min_step), (min_x, max_y, max_step),
            (max_x, min_y, min_step), (max_x, min_y, max_step),
            (max_x, max_y, min_step), (max_x, max_y, max_step),
        ]
        edges = [
            (0,1), (0,2), (0,4), (7,3), (7,5), (7,6),
            (1,3), (1,5), (2,3), (2,6), (4,5), (4,6)
        ]
        for (i, j) in edges:
            p1, p2 = corners[i], corners[j]
            xs, ys, zs = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
            ax.plot(xs, ys, zs)

    plt.title("3D Bounding Boxes by label_trace_id")
    plt.show() 