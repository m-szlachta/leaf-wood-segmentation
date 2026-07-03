import datetime
import numpy as np
import open3d as o3d
import networkx as nx
from Graph_Path import array_to_graph, extract_path_info
from LS_circle import getRootPt
from ExtractInitWood import extract_init_wood
from ExtractFinalWood import extract_final_wood
from PostProcess import smooth_labels, fill_trunk
from Accuracy_evaluation import evaluate_indicators
from Visualization import show_graph, sp_graph, show_pcd
import laspy
import os


class WL_Segmentation():
    def __init__(self, wood_label, leafs_label):
        self.wood_label = wood_label
        self.leafs_label = self.leafs_label
        os.environ.pop("WAYLAND_DISPLAY", None)   # force GLFW onto X11/XWayland
        os.environ["XDG_SESSION_TYPE"] = "x11"

    @staticmethod
    def get_paths(inputh_path):
        paths = []
        for subdir, dirs, files in os.walk(inputh_path):
            for file in files:
                paths.append(os.path.join(subdir, file))
        print(paths)



    @staticmethod
    def _las_to_pcd(las_path):
        las = laspy.read(las_path)

        points = np.vstack((las.x, las.y, las.z)).T
        offset = points.min(axis=0)
        points_local = points - offset

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_local)

        pcd = np.asarray(pcd.points)

    def segmentation():
        pass

    def post_processing():
        pass
    
    def save_to_las(self, output_path, pcd, labels):
        header = laspy.LasHeader(point_format=2, version="1.2")  

        new_las = laspy.LasData(header)
        new_las.x = pcd[:, 0]
        new_las.y = pcd[:, 1]
        new_las.z = pcd[:, 2]
        new_las.classification = labels

        rgb8 = np.where(
            (labels == self.wood_label)[:, None],
            np.array([160, 82, 45], dtype=np.uint16),   # wood
            np.array([60, 170, 70], dtype=np.uint16),   # leaf
        )
        rgb16 = (rgb8 * 257).astype(np.uint16)
        new_las.red, new_las.green, new_las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
        new_las.write(output_path)


if __name__ == "__main__":
    WL_Segmentation.get_paths('data/instance_seg_forest')