#!/usr/bin/env python3
"""Modified from https://github.com/PRBonn/lidar-bonnetal/blob/master/train/common/laserscan.py"""
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d # type: ignore
from scipy.ndimage import median_filter
from sklearn.neighbors import NearestNeighbors


class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-24.9):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                    dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                        dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                    dtype=np.int32)       # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]    # get xyz
        remissions = scan[:, 3]  # get remission

        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points    # get xyz
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in radians
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)  # total field of view in radians

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / (depth + 1e-8))

        # get projections in image coords (scaled to [0, 1])
        proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W                              # in [0.0, W]
        proj_y *= self.proj_H                              # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0, W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in original order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0, H-1]
        self.proj_y = np.copy(proj_y)  # store a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assign to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)

        return
    
    def undo_projection(self, range_image):
        """Inverse projection: reconstruct 3D point cloud from the range image."""
        
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in radians
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)             # total field of view in radians

        # Generate image grid
        u = np.linspace(0, self.proj_W - 1, self.proj_W)  # horizontal indices
        v = np.linspace(0, self.proj_H - 1, self.proj_H)  # vertical indices
        grid_u, grid_v = np.meshgrid(u, v)

        # Convert image indices back to normalized coordinates
        proj_x = grid_u / self.proj_W  # normalized [0, 1]
        proj_y = grid_v / self.proj_H  # normalized [0, 1]

        # Revert the normalization to angles
        yaw = -(proj_x * 2.0 - 1.0) * np.pi  # yaw in [-pi, pi]
        pitch = (1.0 - proj_y) * fov - abs(fov_down)  # pitch in [fov_down, fov_up]

        # Retrieve depth values from the range image
        depth = range_image[grid_v.astype(int), grid_u.astype(int)]

        # Convert spherical coordinates back to Cartesian coordinates
        scan_x = depth * np.cos(pitch) * np.cos(yaw)
        scan_y = depth * np.cos(pitch) * np.sin(yaw)
        scan_z = depth * np.sin(pitch)

        # Stack into a point cloud (N x 3)
        point_cloud = np.stack((scan_x, scan_y, scan_z), axis=-1).reshape(-1, 3)

        valid_mask = (depth > 0) & np.isfinite(depth)
        point_cloud = point_cloud[valid_mask.ravel()]

        return point_cloud
    
    def compute_normals_from_points(self, k=10):
        """
        Computes 3D normals for the point cloud using k-nearest neighbors.
        Args:
            k: Number of nearest neighbors to consider for normal computation.
        Returns:
            normals: A numpy array of shape (N, 3) containing normals for each point.
        """
        if self.points is None or self.points.shape[0] == 0:
            raise ValueError("Point cloud is empty. Load or set a point cloud first.")

        # Initialize array for normals
        normals = np.zeros_like(self.points)

        # Use a k-d tree for efficient neighbor search
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(self.points)
        _, indices = nbrs.kneighbors(self.points)

        for i, neighbors_idx in enumerate(indices):
            # Exclude the point itself (first neighbor)
            neighbors = self.points[neighbors_idx[1:]]  # Exclude the first neighbor (itself)

            # Compute covariance matrix of neighbors
            cov_matrix = np.cov(neighbors, rowvar=False)

            # Eigen decomposition of the covariance matrix
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)

            # Normal is the eigenvector corresponding to the smallest eigenvalue
            normal = eigvecs[:, 0]

            # Ensure consistency of the normal direction
            if np.dot(normal, self.points[i]) > 0:
                normal = -normal

            # Store the normal
            normals[i] = normal

        return normals
    
    def compute_normals_range_image(self, k=10):
        """
        Computes a normals range image by projecting the 3D normals back onto the range image.
        """
        normals = self.compute_normals_from_points(k=k)

        # Initialize the normals range image
        normals_range_image = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)

        for i in range(self.points.shape[0]):
            px, py = self.proj_x[i], self.proj_y[i]
            normals_range_image[py, px] = normals[i]

        return normals_range_image

    def fill_holes_median(self, range_image, kernel_size=5):
        
        if np.shape(range_image)[-1] == 3:
            filled_range_image = np.zeros(np.shape(range_image))
            filled_range_image[:,:,0] = median_filter(range_image[:,:,0], size=kernel_size)
            filled_range_image[:,:,1] = median_filter(range_image[:,:,1], size=kernel_size)
            filled_range_image[:,:,2] = median_filter(range_image[:,:,2], size=kernel_size)
        else:
            filled_range_image = median_filter(range_image, size=kernel_size)
        return filled_range_image

    def visualize_projection(self):
        plt.imshow(self.proj_range)
        plt.show()
        
    def visualize_laser_scan(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        o3d.visualization.draw_geometries([pcd])


class SemLaserScan(LaserScan):
    """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
    EXTENSIONS_LABEL = ['.label']

    def __init__(self, sem_color_dict=None, project=False, H=64, W=2048, fov_up=3.0, fov_down=-24.9):
        super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
        self.reset()

        # make semantic colors
        max_sem_key = 0
        for key, data in sem_color_dict.items():
            if key + 1 > max_sem_key:
                max_sem_key = key + 1
        self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for key, value in sem_color_dict.items():
            self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

        # make instance colors
        max_inst_id = 100000
        self.inst_color_lut = np.random.uniform(low=0.0,
                                                high=1.0,
                                                size=(max_inst_id, 3))
        # force zero to a gray-ish color
        self.inst_color_lut[0] = np.full((3), 0.1)
        
    def reset(self):
        """ Reset scan members. """
        super(SemLaserScan, self).reset()

        # semantic labels
        self.sem_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
        self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # instance labels
        self.inst_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
        self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # projection color with semantic labels
        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                        dtype=np.int32)              # [H,W]  label
        self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                        dtype=float)              # [H,W,3] color

        # projection color with instance labels
        self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                        dtype=np.int32)              # [H,W]  label
        self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                        dtype=float)              # [H,W,3] color

    def open_label(self, filename):
        """ Open raw scan and fill in attributes
        """
        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
            raise RuntimeError("Filename extension is not valid label file.")

        # if all goes well, open label
        label = np.fromfile(filename, dtype=np.uint32)
        label = label.reshape((-1))

        # set it
        self.set_label(label)

    def set_label(self, label):
        """ Set points for label not from file but from np
        """
        # check label makes sense
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")

        # only fill in attribute if the right size
        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label & 0xFFFF  # semantic label in lower half
            self.inst_label = label >> 16    # instance id in upper half
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        # sanity check
        assert((self.sem_label + (self.inst_label << 16) == label).all())

        if self.project:
            self.do_label_projection()

    def colorize(self):
        """ Colorize pointcloud with the color of each semantic label
        """
        self.sem_label_color = self.sem_color_lut[self.sem_label]
        self.sem_label_color = self.sem_label_color.reshape((-1, 3))

        self.inst_label_color = self.inst_color_lut[self.inst_label]
        self.inst_label_color = self.inst_label_color.reshape((-1, 3))

    def do_label_projection(self):
        # only map colors to labels that exist
        mask = self.proj_idx >= 0

        # semantics
        self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
        self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

        # instances
        self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
        self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]
    
    def learning_map(self, learning_map: dict):
        ## Project sem_label to the learning map
        self.proj_sem_label = np.vectorize(lambda x: learning_map.get(x, x))(self.proj_sem_label)
        return
    
    def inv_learning_map(self, learning_map_inv: dict):
        ## Project sem_label to the learning map
        self.proj_sem_label = np.vectorize(lambda x: learning_map_inv.get(x, x))(self.proj_sem_label)
        return

    def map_semantic_classes(self, learning_map:dict, learning_map_inv:dict):
        ## Map to the learning map inverse
        self.sem_label = np.vectorize(lambda x: learning_map.get(x, x))(self.sem_label)
        self.sem_label = np.vectorize(lambda x: learning_map_inv.get(x, x))(self.sem_label)
    
    def colorize_sem_image(self, color_map: dict):
        ## Colorize the semantic image
        self.sem_label_color = np.vectorize(lambda x: color_map.get(x, (0, 0, 0)))(self.proj_sem_label)
        return
    
    def knn_label_projection(self, range_image: np.ndarray,
                            sem_image: np.ndarray,
                            k: int = 1) -> (np.ndarray, np.ndarray):
        """
        Unproject a 2D predicted range image + semantic labels into 3D,
        then optionally apply a 3D k-NN majority vote to refine labels.

        Args:
            range_image:  (H, W) array with predicted range/depth. 
            sem_image:    (H, W) array with predicted semantic class IDs in [0..num_classes-1].
            undo_projection: a function that unprojects a range image -> (N,3) 3D points.
                                e.g., your 'undo_projection' method
            k:            number of neighbors for smoothing. 
                        - If k=1 => direct labeling (no 3D smoothing).
                        - If k>1 => majority vote among neighbors.

        Returns:
            points_3d:  (N, 3) unprojected 3D points
            labels_3d:  (N,) refined semantic label for each 3D point
        """
        point_cloud = self.undo_projection(range_image)  # shape (N,3)
        H, W = range_image.shape
        assert H == self.proj_H and W == self.proj_W, (
            f"Range image shape {H}x{W} doesn't match expected {self.proj_H}x{self.proj_W}!"
        )

        # Rebuild coordinate grid
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))  # (W,H) -> (H,W)
        u_grid = u_grid.astype(int)
        v_grid = v_grid.astype(int)

        # Flatten (row-major order)
        range_flat = range_image[v_grid, u_grid].reshape(-1)
        sem_flat   = sem_image[v_grid, u_grid].reshape(-1)

        # Determine valid points (the same as in your undo_projection)
        valid = (range_flat > 0) & np.isfinite(range_flat)

        # Extract valid semantic labels
        sem_valid = sem_flat[valid]
        if point_cloud.shape[0] != sem_valid.shape[0]:
            raise ValueError(
                f"Mismatch in valid points: {point_cloud.shape[0]} vs {sem_valid.shape[0]}."
                "Ensure your undo_projection matches the exact logic for filtering invalid depths."
            )

        if k == 1:
            return point_cloud, sem_valid

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(point_cloud)
        _, indices = nbrs.kneighbors(point_cloud)

        labels_3d = np.zeros_like(sem_valid)
        for i in range(point_cloud.shape[0]):
            neighbor_labels = sem_valid[indices[i]]  # Labels of the k neighbors
            vals, counts = np.unique(neighbor_labels, return_counts=True)
            majority_label = vals[np.argmax(counts)]
            labels_3d[i] = majority_label

        return point_cloud, labels_3d
        
    def visualize_label_laser_scan(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        ## Colors are in bgr format
        pcd.colors = o3d.utility.Vector3dVector(self.sem_label_color[:, ::-1])
        o3d.visualization.draw_geometries([pcd])
        
    def visualize_instance_label_laser_scan(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        ## Colors are in bgr format
        pcd.colors = o3d.utility.Vector3dVector(self.inst_label_color[:, ::-1])
        o3d.visualization.draw_geometries([pcd])
