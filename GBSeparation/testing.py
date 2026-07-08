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
    def __init__(self, wood_label:int = 0, leafs_label: int =1):
        self.wood_label = wood_label
        self.leafs_label = leafs_label
        os.environ.pop("WAYLAND_DISPLAY", None)   # force GLFW onto X11/XWayland
        os.environ["XDG_SESSION_TYPE"] = "x11"

    @staticmethod
    def get_paths(input_path: str) -> str:
        paths = []

        for subdir, dirs, files in os.walk(input_path):
            print(f'{subdir}, {dirs}, {files}')
            for file in files:
                paths.append(os.path.join(subdir, file))
            return paths

        return [input_path]

    @staticmethod
    def _las_to_pcd(las_path: str) -> tuple[np.ndarray, np.ndarray]:
        las = laspy.read(las_path)

        points = np.vstack((las.x, las.y, las.z)).T
        offset = points.min(axis=0)
        points_local = points - offset

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_local)

        pcd = np.asarray(pcd.points)
        return pcd, offset

    def segmentation(self, pcd: np.ndarray, t_linearity: float = 0.9, t_error: float = 0.1, curvy_treshold: float = 0.02, max_trunk_radius: float = 0.5) -> np.ndarray:
        treeHeight = np.max(pcd[:, 2])-np.min(pcd[:, 2])

        self.root, fit_seg, self.trunk_radius = getRootPt(pcd, lower_h=0.0, upper_h=0.2)

        if self.trunk_radius > max_trunk_radius:
            print(f"trunk_radius {self.trunk_radius:.3f} m implausible, clamping to {max_trunk_radius} m")
            self.trunk_radius = max_trunk_radius
        pcd = np.append(pcd, self.root, axis=0)
        root_id = pcd.shape[0]-1

        G = array_to_graph(pcd, root_id, kpairs=3, knn=300, nbrs_threshold=treeHeight/30, nbrs_threshold_step=treeHeight/60)

        path_dis, path_list = extract_path_info(G, root_id, return_path=True)

        init_wood_ids = extract_init_wood(pcd, G, root_id, path_dis, path_list,
                                  split_interval=[0.1, 0.2, 0.3, 0.5, 1], max_angle=0.25*np.pi,
                                  t_linearity=t_linearity, t_error=t_error,
                                  curve_threshold=curvy_treshold)

        final_wood_mask = extract_final_wood(pcd, root_id, path_dis, path_list, init_wood_ids, G)

        return final_wood_mask

    def post_processing(
            self,
            pcd: np.ndarray,
            final_wood_mask: np.ndarray,
            smooth_k: int = 9,
            smooth_iters: int = 1,
            fill_trunk_flag: bool = True,
            trunk_radius_factor: float = 3.0,
            trunk_spread_factor: float = 3.0,
        ) -> tuple[np.ndarray, np.ndarray]:

        if smooth_k > 0:
            final_wood_mask = smooth_labels(pcd, final_wood_mask, k=smooth_k, iters=smooth_iters)

        if fill_trunk_flag:
            final_wood_mask, crown_base_z = fill_trunk(pcd, final_wood_mask, self.root[0, :2], self.trunk_radius,
                                            radius_factor=trunk_radius_factor,
                                            spread_factor=trunk_spread_factor)

        final_wood_mask[-1] = False
        wood = pcd[final_wood_mask]
        final_wood_mask[-1] = True
        leaf = pcd[~final_wood_mask]

        wood_labels = np.full(len(wood), self.wood_label)
        leaf_labels = np.full(len(leaf), self.leafs_label)
        labels = np.hstack((wood_labels, leaf_labels)).T

        final_pcd = np.vstack((wood, leaf))
        return final_pcd, labels

    def save_to_las(self, pcd: np.ndarray, labels: np.ndarray, output_path: str):
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

    def run_segmentation_pipeline(
            self,
            input_path: str,
            output_path: str,
            t_linearity: float = 0.85,
            t_error: float = 0.1,
            curvy_treshold: float = 0.02,
            smooth_k: int = 13,
            smooth_iters: int = 2,
            fill_trunk: bool = True,
            trunk_radius_factor: float = 2.5,
            trunk_spread_factor: float = 3.0,
        ):
            paths = self.get_paths(input_path)

            for path in paths:
                print(f"starting segmentation {path}")
                pcd, offset = self._las_to_pcd(path)
                final_wood_mask = self.segmentation(pcd, t_linearity, t_error, curvy_treshold)
                final_pcd, labels = self.post_processing(
                    pcd,
                    final_wood_mask,
                    smooth_k,
                    smooth_iters,
                    fill_trunk,
                    trunk_radius_factor,
                    trunk_spread_factor,
                )
                final_pcd = final_pcd + offset
                final_path = os.path.join(output_path, os.path.basename(path))
                self.save_to_las(final_pcd, labels, final_path)

if __name__ == "__main__":
    seg = WL_Segmentation()
    seg.run_segmentation_pipeline(input_path='data/instance_seg_forest', output_path='data/wyniki')
