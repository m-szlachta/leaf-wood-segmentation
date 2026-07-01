import laspy
import open3d as o3d
import numpy as np

LAS_PATH = 'data/ITWL_Grajewo20_mini_mod_2.laz'

las = laspy.read(LAS_PATH)

points = np.vstack((las.x, las.y, las.z)).T

# LAS coordinates are in a national grid (large values, e.g. ~5.9e6). Open3D
# stores .pcd point coordinates as 32-bit floats (~7 significant digits), so
# writing the georeferenced values directly loses sub-meter precision and the
# points appear rounded. Shift to a local origin so the stored values are small
# and keep full precision. Save the offset to restore absolute coordinates later.
offset = points.min(axis=0)
points_local = points - offset

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_local)

o3d.io.write_point_cloud('data/single_tree.pcd', pcd)