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
    def __init__(
        self,
        wood_label: int = 0,
        leafs_label: int = 1,
        t_linearity: float = 0.85,
        t_error: float = 0.1,
        curvy_treshold: float = 0.02,
        max_trunk_radius: float = 0.5,
        smooth_k: int = 13,
        smooth_iters: int = 2,
        fill_trunk_flag: bool = True,
        trunk_radius_factor: float = 2.5,
        trunk_spread_factor: float = 3.0,
    ):
        self.wood_label = wood_label
        self.leafs_label = leafs_label
        self.t_linearity = t_linearity
        self.t_error = t_error
        self.curvy_treshold = curvy_treshold
        self.max_trunk_radius = max_trunk_radius
        self.smooth_k = smooth_k
        self.smooth_iters = smooth_iters
        self.fill_trunk_flag = fill_trunk_flag
        self.trunk_radius_factor = trunk_radius_factor
        self.trunk_spread_factor = trunk_spread_factor

        os.environ.pop("WAYLAND_DISPLAY", None)   # force GLFW onto X11/XWayland
        os.environ["XDG_SESSION_TYPE"] = "x11"

    def _segmentation(self, pcd: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        treeHeight = np.max(pcd[:, 2])-np.min(pcd[:, 2])

        self.root, fit_seg, self.trunk_radius = getRootPt(pcd, lower_h=0.0, upper_h=0.2)

        if self.trunk_radius > self.max_trunk_radius:
            print(f"trunk_radius {self.trunk_radius:.3f} m implausible, clamping to {self.max_trunk_radius} m")
            self.trunk_radius = self.max_trunk_radius
        pcd = np.append(pcd, self.root, axis=0)
        root_id = pcd.shape[0]-1

        G = array_to_graph(pcd, root_id, kpairs=3, knn=300, nbrs_threshold=treeHeight/30, nbrs_threshold_step=treeHeight/60)

        path_dis, path_list = extract_path_info(G, root_id, return_path=True)

        init_wood_ids = extract_init_wood(pcd, G, root_id, path_dis, path_list,
                                  split_interval=[0.1, 0.2, 0.3, 0.5, 1], max_angle=0.25*np.pi,
                                  t_linearity=self.t_linearity, t_error=self.t_error,
                                  curve_threshold=self.curvy_treshold)

        final_wood_mask = extract_final_wood(pcd, root_id, path_dis, path_list, init_wood_ids, G)

        return pcd, final_wood_mask, root_id

    def _post_processing(self, pcd: np.ndarray, final_wood_mask: np.ndarray) -> np.ndarray:
        if self.smooth_k > 0:
            final_wood_mask = smooth_labels(pcd, final_wood_mask, k=self.smooth_k, iters=self.smooth_iters)

        if self.fill_trunk_flag:
            final_wood_mask, crown_base_z = fill_trunk(pcd, final_wood_mask, self.root[0, :2], self.trunk_radius,
                                            radius_factor=self.trunk_radius_factor,
                                            spread_factor=self.trunk_spread_factor)

        return final_wood_mask

    def segment(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        xyz = points[:, :3].astype(float)
        n = xyz.shape[0]

        extended_xyz, final_wood_mask, root_id = self._segmentation(xyz)
        final_wood_mask = self._post_processing(extended_xyz, final_wood_mask)

        wood = final_wood_mask[:n]   # drop the internally-appended root, keep input order
        labels = np.where(wood, self.wood_label, self.leafs_label)

        return np.hstack((points, labels[:, None]))

# code for testing

def get_paths(input_path: str) -> list[str]:
    paths = []

    for subdir, dirs, files in os.walk(input_path):
        print(f'{subdir}, {dirs}, {files}')
        for file in files:
            paths.append(os.path.join(subdir, file))
        return paths

    return [input_path]


def las_to_pcd(las_path: str) -> tuple[np.ndarray, np.ndarray]:
    las = laspy.read(las_path)

    points = np.vstack((las.x, las.y, las.z)).T
    offset = points.min(axis=0)
    points_local = points - offset

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_local)

    pcd = np.asarray(pcd.points)
    return pcd, offset


def save_to_las(pcd: np.ndarray, labels: np.ndarray, output_path: str, wood_label: int = 0):
    header = laspy.LasHeader(point_format=2, version="1.2")

    labels = np.asarray(labels).astype(np.uint8)

    new_las = laspy.LasData(header)
    new_las.x = pcd[:, 0]
    new_las.y = pcd[:, 1]
    new_las.z = pcd[:, 2]
    new_las.classification = labels

    rgb8 = np.where(
        (labels == wood_label)[:, None],
        np.array([160, 82, 45], dtype=np.uint16),   # wood
        np.array([60, 170, 70], dtype=np.uint16),   # leaf
    )
    rgb16 = (rgb8 * 257).astype(np.uint16)
    new_las.red, new_las.green, new_las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]

    new_las.write(output_path)


if __name__ == "__main__":
    input_path = 'data/instance_seg_forest'
    output_path = 'data/wyniki1'
    seg = WL_Segmentation()

    for path in get_paths(input_path):
        print(f"starting segmentation {path}")
        pcd, offset = las_to_pcd(path)
        out = seg.segment(pcd)
        points, labels = out[:, :3] + offset, out[:, 3]
        final_path = os.path.join(output_path, os.path.basename(path))
        save_to_las(points, labels, final_path, wood_label=seg.wood_label)
