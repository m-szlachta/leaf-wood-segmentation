import datetime
import numpy as np
import open3d as o3d
import networkx as nx
from Graph_Path import array_to_graph, extract_path_info
from LS_circle import getRootPt
from ExtractInitWood import extract_init_wood
from ExtractFinalWood import extract_final_wood
from Accuracy_evaluation import evaluate_indicators
from Visualization import show_graph, sp_graph, show_pcd
import laspy
import os

os.environ.pop("WAYLAND_DISPLAY", None)   # force GLFW onto X11/XWayland
os.environ["XDG_SESSION_TYPE"] = "x11"

INPUT_PATH = 'data/tree_v1.laz'
OUTPUT_PATH = 'data/segmented_tree_v1.las'

# --- Wood-cluster classification thresholds (tune these) ---
# A cluster is classified as "cylinder wood" if its 2D circle-fit relative RMS error
# is below T_ERROR and its curvature (thickness) exceeds CURVE_THRESHOLD; otherwise it
# is "linear wood" if its linearity exceeds T_LINEARITY. Lower T_ERROR / higher
# CURVE_THRESHOLD = stricter = less leaf mislabelled as wood.
T_LINEARITY = 0.95      # linear-shape threshold (was effectively 0.90)
T_ERROR = 0.10          # cylinder-fit relative error threshold (was effectively 0.50)
CURVE_THRESHOLD = 0.03  # minimum cross-sectional curvature for a cylinder (was 0.01)

las = laspy.read(INPUT_PATH)
points = np.vstack((las.x, las.y, las.z)).T
offset = points.min(axis=0)
points_local = points - offset

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_local)

pcd = np.asarray(pcd.points)

# Please ensure that the growth direction of the tree is parallel to the Z coordinate axis.
treeHeight = np.max(pcd[:, 2])-np.min(pcd[:, 2])

# fit the root point.
root, fit_seg = getRootPt(pcd, lower_h=0.0, upper_h=0.2)
pcd = np.append(pcd, root, axis=0)
root_id = pcd.shape[0]-1
print("root_ID:", root_id)
#show_pcd(pcd)

# construct networkx Graph.
print(str(datetime.datetime.now()) + ' | >>>constructing networkx Graph...')
G = array_to_graph(pcd, root_id, kpairs=3, knn=300, nbrs_threshold=treeHeight/30, nbrs_threshold_step=treeHeight/60)

# # save/read already constructed Graph to reduce processing time.
# nx.write_gpickle(G, 'E:\\folder\\G.gpickle')
# G = nx.read_gpickle('E:\\folder\\G.gpickle')

print(">>>connected components of constructed Graph: ", nx.number_connected_components(G))
#show_graph(pcd, G)

# extract path info information from graph
print(str(datetime.datetime.now()) + ' | >>>extracting shortest path information...')
path_dis, path_list = extract_path_info(G, root_id, return_path=True)
# show_graph(pcd, sp_graph(path_list, root_id))

# extract initial wood points.
print(str(datetime.datetime.now()) + ' | >>>extracting initial wood points...')
init_wood_ids = extract_init_wood(pcd, G, root_id, path_dis, path_list,
                                  split_interval=[0.1, 0.2, 0.3, 0.5, 1], max_angle=0.25*np.pi,
                                  t_linearity=T_LINEARITY, t_error=T_ERROR,
                                  curve_threshold=CURVE_THRESHOLD)

# extract final wood points.
print(str(datetime.datetime.now()) + ' | >>>extracting final wood points...')
final_wood_mask = extract_final_wood(pcd, root_id, path_dis, path_list, init_wood_ids, G)

# remove the inserted root point and extract wood/leaf points by mask index.
final_wood_mask[-1] = False
wood = pcd[final_wood_mask]
final_wood_mask[-1] = True
leaf = pcd[~final_wood_mask]

show_pcd(wood)
show_pcd(leaf)

WOOD_LABEL = 0
LEAF_LABEL = 1


wood_labels = np.full(len(wood), WOOD_LABEL)
leaf_labels = np.full(len(leaf), LEAF_LABEL)
labels = np.hstack((wood_labels, leaf_labels)).T

final_pcd = np.vstack((wood, leaf))

header = laspy.LasHeader(point_format=2, version="1.2")  

new_las = laspy.LasData(header)
new_las.x = final_pcd[:, 0]
new_las.y = final_pcd[:, 1]
new_las.z = final_pcd[:, 2]
new_las.classification = labels

rgb8 = np.where(
    (labels == WOOD_LABEL)[:, None],
    np.array([160, 82, 45], dtype=np.uint16),   # wood
    np.array([60, 170, 70], dtype=np.uint16),   # leaf
)
rgb16 = (rgb8 * 257).astype(np.uint16)
new_las.red, new_las.green, new_las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
new_las.write(OUTPUT_PATH)
