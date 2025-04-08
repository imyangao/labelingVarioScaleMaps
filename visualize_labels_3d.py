import geopandas as gpd
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon
import numpy as np
from sqlalchemy import create_engine
import pandas as pd

def connect_to_db():
    """Connect to the PostgreSQL database using SQLAlchemy."""
    # URL encode the special characters in the password
    password = "Gy%40001130"  # @ is encoded as %40
    return create_engine(f'postgresql://postgres:{password}@localhost:5432/tgap_test')

def get_faces_and_labels():
    """Get faces and their labels from the database."""
    engine = connect_to_db()

    # faces_sql = """
    # SELECT DISTINCT la.face_id, la.step_value, f.step_low, f.step_high, f.feature_class,
    #        ST_Force2D(la.face_geom) AS geometry
    # FROM label_anchors la
    # JOIN yan_tgap_face f ON f.face_id = la.face_id
    # WHERE (f.feature_class >= 10000 AND f.feature_class < 11000)  -- Roads
    #    OR (f.feature_class >= 12000 AND f.feature_class < 13000)  -- Water
    #    OR (f.feature_class >= 13000 AND f.feature_class < 14000)  -- Buildings
    # AND ST_IsValid(la.face_geom)
    # LIMIT 5000;
    # """
    faces_sql = """
    SELECT DISTINCT
        la.face_id,
        la.step_value,
        f.step_low,
        f.step_high,
        f.feature_class,
        ST_Force2D(la.face_geom) AS geometry
    FROM label_anchors la
    JOIN yan_tgap_face f
        ON f.face_id = la.face_id
    WHERE
        la.fits IS TRUE
        AND (
              (f.feature_class >= 10000 AND f.feature_class < 11000)  -- roads
           OR (f.feature_class >= 12000 AND f.feature_class < 13000)  -- water
           OR (f.feature_class >= 13000 AND f.feature_class < 14000)  -- buildings
        )
        AND ST_IsValid(la.face_geom);
    """
    
    faces_df = gpd.read_postgis(faces_sql, engine, geom_col='geometry')

    print(f"Number of faces loaded: {len(faces_df)}")
    print("\nFace geometry types:")
    print(faces_df['geometry'].geom_type.value_counts())
    print("\nSample face data:")
    print(faces_df.head())
    
    # Get labels that fit inside polygons
    labels_sql = """
    SELECT l.label_id, l.face_id, l.step_value, l.feature_class, 
           l.anchor_geom, l.angle, l.fits, l.name, l.label_trace_id
    FROM label_anchors l
    WHERE l.fits = true;
    """
    
    labels_df = gpd.read_postgis(labels_sql, engine, geom_col='anchor_geom')

    bounds_sql = """
    SELECT
        label_trace_id,
        min_x,
        max_x,
        min_y,
        max_y,
        min_step,
        max_step
    FROM label_trace_3d_bounds;
    """

    bounds_df = pd.read_sql(bounds_sql, engine)
    
    return faces_df, labels_df, bounds_df


def add_bounding_box_trace(fig, min_x, max_x, min_y, max_y, min_step, max_step, name='Bounding Box'):
    """
    Adds a 3D wireframe bounding box to the figure.
    """
    # Corner points of the box
    corners = [
        (min_x, min_y, min_step),
        (min_x, min_y, max_step),
        (min_x, max_y, min_step),
        (min_x, max_y, max_step),
        (max_x, min_y, min_step),
        (max_x, min_y, max_step),
        (max_x, max_y, min_step),
        (max_x, max_y, max_step),
    ]

    # Edges: each tuple is a pair of corner indices
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (4, 6), (5, 7), (6, 7),
    ]

    # Add each edge as a small "line"
    for i, j in edges:
        x_coords = [corners[i][0], corners[j][0]]
        y_coords = [corners[i][1], corners[j][1]]
        z_coords = [corners[i][2], corners[j][2]]

        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            line=dict(
                color='magenta',  # choose any color
                width=2
            ),
            name=name,
            showlegend=False  # set True if you want a legend entry for each edge
        ))


def add_solid_bounding_box_trace(
        fig,
        min_x,
        max_x,
        min_y,
        max_y,
        min_step,
        max_step,
        name='Bounding Box',
        color='magenta'
):
    """
    Adds a 3D solid bounding box (cuboid) as a Mesh3d with partial transparency.
    """
    import numpy as np
    import plotly.graph_objects as go

    # Define the 8 corner points
    corners = np.array([
        [min_x, min_y, min_step],
        [min_x, min_y, max_step],
        [min_x, max_y, min_step],
        [min_x, max_y, max_step],
        [max_x, min_y, min_step],
        [max_x, min_y, max_step],
        [max_x, max_y, min_step],
        [max_x, max_y, max_step]
    ])

    x = corners[:, 0]
    y = corners[:, 1]
    z = corners[:, 2]

    # The faces of a cuboid can be defined by two triangles each.
    # One simple approach: define the 12 triangles (i, j, k)
    # that form the 6 rectangular faces.
    # For example, the first face is formed by corners 0,1,2,3, etc.
    # For your convenience, here’s a standard indexing for the cuboid:
    I = [0, 0, 0, 1, 2, 4, 7, 3, 5, 6, 6, 2]
    J = [2, 3, 1, 2, 4, 7, 6, 5, 7, 2, 3, 6]
    K = [3, 1, 2, 4, 7, 6, 5, 7, 3, 6, 6, 3]

    fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=I,
        j=J,
        k=K,
        color=color,  # e.g. 'rgba(255,0,255,1)' for fully opaque
        opacity=0.2,  # 0.2 means 20% opaque / 80% transparent
        name=name,
        hovertemplate=(
            f"{name}<br>"
            f"x-range: [{min_x}, {max_x}]<br>"
            f"y-range: [{min_y}, {max_y}]<br>"
            f"step-range: [{min_step}, {max_step}]<extra></extra>"
        )
    ))


def create_3d_visualization(faces_df, labels_df, bounds_df):
    """Create a 3D visualization of faces and labels rendering."""
    fig = go.Figure()
    
    # Pre-calculate colors for feature classes
    colors = {
        'roads': 'red',
        'water': 'blue',
        'buildings': 'gray'
    }
    
    # Add loading progress indicator
    total_faces = len(faces_df)
    processed_faces = 0
    
    # Group faces by feature class to reduce number of traces
    for feature_class, group in faces_df.groupby('feature_class'):
        print(f"\nProcessing feature class {feature_class} with {len(group)} faces")
        
        # Determine color based on feature class
        if 10000 <= feature_class < 11000:
            color = colors['roads']
        elif 12000 <= feature_class < 13000:
            color = colors['water']
        else:  # buildings
            color = colors['buildings']
        
        # Process all polygons in the group at once
        for idx, face in group.iterrows():
            if face['geometry'] is None:
                continue
                
            # Handle GeometryCollection
            if face['geometry'].geom_type == 'GeometryCollection':
                if face['geometry'].is_empty:
                    continue
                    
                # Process each geometry in the collection
                for geom in face['geometry'].geoms:
                    if geom.geom_type == 'Polygon':
                        coords = list(geom.exterior.coords)
                        if len(coords) < 3:
                            continue
                            
                        # # Simplify polygon if too many points
                        # if len(coords) > 100:
                        #     coords = coords[::2]  # Take every other point
                            
                        x_coords = [coord[0] for coord in coords]
                        y_coords = [coord[1] for coord in coords]
                        
                        fig.add_trace(go.Mesh3d(
                            x=x_coords,
                            y=y_coords,
                            z=[face['step_value']] * len(coords),
                            i=[0] * (len(coords) - 2),
                            j=list(range(1, len(coords) - 1)),
                            k=list(range(2, len(coords))),
                            color=color,
                            opacity=0.5,
                            name=f'Face {face["face_id"]} ({feature_class})',
                            hovertemplate=(
                                f"Face ID: {face['face_id']}<br>" +
                                f"Feature Class: {feature_class}<br>" +
                                f"Step: {face['step_value']}<br>" +
                                f"Step Range: {face['step_low']} - {face['step_high']}<br>" +
                                "<extra></extra>"
                            )
                        ))
            elif face['geometry'].geom_type == 'Polygon':
                coords = list(face['geometry'].exterior.coords)
                if len(coords) < 3:
                    continue
                    
                # # Simplify polygon if too many points
                # if len(coords) > 100:
                #     coords = coords[::2]  # Take every other point
                    
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]
                
                fig.add_trace(go.Mesh3d(
                    x=x_coords,
                    y=y_coords,
                    z=[face['step_value']] * len(coords),
                    i=[0] * (len(coords) - 2),
                    j=list(range(1, len(coords) - 1)),
                    k=list(range(2, len(coords))),
                    color=color,
                    opacity=0.5,
                    name=f'Face {face["face_id"]} ({feature_class})',
                    hovertemplate=(
                        f"Face ID: {face['face_id']}<br>" +
                        f"Feature Class: {feature_class}<br>" +
                        f"Step: {face['step_value']}<br>" +
                        f"Step Range: {face['step_low']} - {face['step_high']}<br>" +
                        "<extra></extra>"
                    )
                ))
            elif face['geometry'].geom_type == 'MultiPolygon':
                # Process each polygon in the MultiPolygon
                for polygon in face['geometry'].geoms:
                    coords = list(polygon.exterior.coords)
                    if len(coords) < 3:
                        continue
                        
                    # # Simplify polygon if too many points
                    # if len(coords) > 100:
                    #     coords = coords[::2]  # Take every other point
                        
                    x_coords = [coord[0] for coord in coords]
                    y_coords = [coord[1] for coord in coords]
                    
                    fig.add_trace(go.Mesh3d(
                        x=x_coords,
                        y=y_coords,
                        z=[face['step_value']] * len(coords),
                        i=[0] * (len(coords) - 2),
                        j=list(range(1, len(coords) - 1)),
                        k=list(range(2, len(coords))),
                        color=color,
                        opacity=0.5,
                        name=f'Face {face["face_id"]} ({feature_class})',
                        hovertemplate=(
                            f"Face ID: {face['face_id']}<br>" +
                            f"Feature Class: {feature_class}<br>" +
                            f"Step: {face['step_value']}<br>" +
                            f"Step Range: {face['step_low']} - {face['step_high']}<br>" +
                            "<extra></extra>"
                        )
                    ))
            
            processed_faces += 1
            if processed_faces % 1000 == 0:
                print(f"Processed {processed_faces}/{total_faces} faces...")
    
    # Group labels by face_id to reduce number of traces
    for face_id, group in labels_df.groupby('face_id'):
        for _, label in group.iterrows():
            # Get label coordinates
            x, y = label['anchor_geom'].x, label['anchor_geom'].y
            
            # Add point with optimized settings
            fig.add_trace(go.Scatter3d(
                x=[x],
                y=[y],
                z=[label['step_value']],
                mode='markers+text',
                marker=dict(
                    size=5,
                    color='black',
                    symbol='circle'
                ),
                text=[label['name'] if label['name'] else f'Label {label["label_id"]}'],
                textposition='top center',
                name=f'Label {label["label_id"]}',
                hovertemplate=(
                    f"Label ID: {label['label_id']}<br>" +
                    f"Name: {label['name'] if label['name'] else 'N/A'}<br>" +
                    f"Face ID: {label['face_id']}<br>" +
                    f"Step: {label['step_value']}<br>" +
                    f"Angle: {label['angle']:.1f}°<br>" +
                    "<extra></extra>"
                )
            ))
            
            # Add rotation indicator
            angle_rad = np.radians(label['angle'])
            length = 100  # Length of rotation indicator
            dx = length * np.cos(angle_rad)
            dy = length * np.sin(angle_rad)
            
            fig.add_trace(go.Scatter3d(
                x=[x, x + dx],
                y=[y, y + dy],
                z=[label['step_value'], label['step_value']],
                mode='lines',
                line=dict(
                    color='black',
                    width=2
                ),
                showlegend=False
            ))

    # Draw movement trajectory lines for label trace groups
    label_traces = labels_df.dropna(subset=['label_trace_id']).groupby(['label_trace_id', 'name'])

    for (trace_id, name), group in label_traces:
        group = group.sort_values('step_value')
        x_vals = group['anchor_geom'].x.values
        y_vals = group['anchor_geom'].y.values
        z_vals = group['step_value'].values

        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='lines',
            line=dict(
                color='green',
                width=3
            ),
            name=f"Trajectory: {name} ({trace_id})",
            hoverinfo='text',
            text=[f"Step: {step}" for step in z_vals],
            showlegend=False  # You can enable this if desired
        ))

        # 1) Identify all relevant trace IDs
        relevant_trace_ids = set(key[0] for key in label_traces.groups.keys())

        # 2) Filter `bounds_df` to only those
        filtered_bounds = bounds_df[bounds_df['label_trace_id'].isin(relevant_trace_ids)]

        # 3) Add bounding boxes only for those traces
        for idx, row in filtered_bounds.iterrows():
            add_bounding_box_trace(
            # add_solid_bounding_box_trace(
                fig,
                row['min_x'],
                row['max_x'],
                row['min_y'],
                row['max_y'],
                row['min_step'],
                row['max_step'],
                name=f"Trace Bounds {row['label_trace_id']}"
            )

    fig.update_layout(
        title='3D Visualization of Faces and Labels',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Step'
        ),
        showlegend=True,
        uirevision=True,  # Preserve zoom/pan state
        hovermode='closest',  # Optimize hover performance
        hoverdistance=100,  # Reduce hover computation
        # Add WebGL rendering
        template='plotly_white'
    )
    
    # Save the figure with optimized settings
    print("Saving visualization...")
    fig.write_html('faces_labels_3d.html', 
                  config={
                      'responsive': True,
                      'scrollZoom': True,
                      'displayModeBar': True,
                      'modeBarButtonsToAdd': ['drawline', 'eraseshape'],
                      'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                      'displaylogo': False,
                      'toImageButtonOptions': {
                          'format': 'png',
                          'width': 1920,
                          'height': 1080,
                          'scale': 2
                      }
                  },
                  include_plotlyjs='cdn',  # Use CDN for faster loading
                  full_html=True)  # Include full HTML for better performance
    
    print("Visualization saved as 'faces_labels_3d.html'")

def main():
    print("Loading data from database...")
    faces_df, labels_df, bounds_df = get_faces_and_labels()
    
    print("Creating 3D visualization...")
    create_3d_visualization(faces_df, labels_df, bounds_df)
    
    print("Done!")

if __name__ == "__main__":
    main() 