import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon, Point

data = [
    (0, Polygon([(0,0),(10,0),(10,5),(0,5)]), Point(5,2)),
    (1, Polygon([(0.5,0.2),(9.5,0.2),(9.5,4.5),(0.5,4.5)]), Point(5,2.5)),
    (2, Polygon([(1,0.5),(9,0.5),(9,4),(1,4)]), Point(5,3)),
]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')  # 3D axes

anchors_x = []
anchors_y = []
anchors_z = []

for (step, poly_shp, anchor) in data:
    # The polygon's exterior ring
    xvals, yvals = poly_shp.exterior.xy
    zvals = [step]*len(xvals)  # same z for all vertices at this step

    # Draw polygon edges in 3D
    ax.plot(xvals, yvals, zvals)

    # Draw anchor
    ax.plot([anchor.x], [anchor.y], [step], marker='o')

    # Collect anchor coords for line
    anchors_x.append(anchor.x)
    anchors_y.append(anchor.y)
    anchors_z.append(step)

# Connect the anchor points in 3D
ax.plot(anchors_x, anchors_y, anchors_z)

ax.set_title("3D Visualization: Polygons Over Time")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Step/Time")
plt.show()
