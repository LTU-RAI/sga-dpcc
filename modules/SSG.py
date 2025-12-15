import os, sys, time, pickle, torch # type: ignore
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np # type: ignore
import networkx as nx # type: ignore
import matplotlib.pyplot as plt # type: ignore
import open3d as o3d # type: ignore
from sgadpcc_utils.LaserScan import SemLaserScan
from torch_geometric.data import Data  # type: ignore


class SSG():
    def __init__(self, vocabulary: dict, layer_4: bool=True, verbose: bool=False):
        self.verbose = verbose
        self.layer_4 = layer_4
        self.vocabulary = vocabulary
        if self.verbose:
            print(f'Vocabulary: {self.vocabulary}')
    
    ## Reset the SSG.
    def reset(self) -> None:
        self.ssg = None
        return
    
    ## Resete scan.
    def reset_scan(self) -> None:
        self.sem_scan = None
        return

    ## Save the SSG to a file.
    def save_ssg(self, path:str) -> None:
        if self.verbose:
            print(f'Saving SSG to: {path}')
        pickle.dump(self.ssg, open(path, 'wb'))
        return

    ## Load the SSG from a file.
    def load_ssg(self, path:str) -> nx.Graph:
        if self.verbose:
            print(f'Loading SSG from: {path}')
        self.ssg = pickle.load(open(path, 'rb'))
        return self.ssg

    ## Convert to float16.
    def to_float16(self) -> None:
        for node in self.ssg.nodes(data=True):
            # if node[1]['points'] is not None:
            #     node[1]['points'] = node[1]['points'].astype(np.float16)
            node[1]['center'] = np.array(node[1]['center'], dtype=np.float16)
            node[1]['extent'] = np.array(node[1]['extent'], dtype=np.float16)
            node[1]['orientation'] = np.array(node[1]['orientation'], dtype=np.float16)
        return

    ## Convert a NetworkX graph to a PyTorch Geometric Data object.
    def nx_to_pyg(self, nx_graph:nx.Graph, kitti_config:dict) -> Data:
        semantic_id = torch.tensor(
            [kitti_config['learning_map'][n[1]['semantic_id']] for n in nx_graph.nodes(data=True)],
            dtype=torch.float
        )
        instance_id = np.array([n[1]['instance_id'] for n in nx_graph.nodes(data=True)])
        num_points = torch.tensor(
            [n[1]['num_points'] for n in nx_graph.nodes(data=True)],
            dtype=torch.float
        )
        edges = list(nx_graph.edges())
        reverse_edges = [(j,i) for i,j in edges]
        all_edges = edges + reverse_edges
        edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        centers = np.array([n[1]['center'] for n in nx_graph.nodes(data=True)])
        extents = np.array([n[1]['extent'] for n in nx_graph.nodes(data=True)])
        orientations = np.array([n[1].get('orientation', np.eye(3)) for n in nx_graph.nodes(data=True)])
        pos = torch.tensor(centers, dtype=torch.float)
        extent = torch.tensor(extents, dtype=torch.float)
        orientation = torch.tensor(orientations, dtype=torch.float)
        points = [n[1]['points'] if n[1]['points'] is not None else torch.zeros((1, 4)) for n in nx_graph.nodes(data=True)]
        latent = [n[1]['latent'] if n[1]['latent'] is not None else torch.zeros((1,)) for n in nx_graph.nodes(data=True)]
        scan_id = nx_graph.graph['scan_id']
        per_layer = nx_graph.graph['per_layer']
        data = Data(
            x=semantic_id.view(-1, 1),
            edge_index=edge_index, 
            instance_id=instance_id, 
            num_points=num_points,
            pos=pos,
            extent=extent,
            orientation=orientation,
            points=points,
            latent=latent,
            scan_id=scan_id, 
            per_layer=per_layer
        )
        return data

    #def pyg_to_nx(self, data:Data) -> nx.Graph:
        pass

    ## Generate the SSG from a SemLaserScan object.
    def generate_ssg(self, sem_scan:SemLaserScan, scan_id:str) -> nx.Graph:
        if self.verbose:
            print(f'Initializing graph for scan: {scan_id}')
        if self.layer_4:
            per_layer = {'terrain':0, 'objects':0, 'agents':0, 'infrastructure':0}
        else:
            per_layer = {'terrain':0, 'objects':0, 'agents':0}
        self.ssg = nx.Graph(scan_id=scan_id, per_layer=per_layer)
        self.sem_scan = sem_scan
        terrain_classes = self.get_terrain_classes(sem_scan)
        if self.layer_4:
            infrastructure_bboxes, infrastructure_classes = self.cluster_infrastructure(sem_scan)
        else:
            infrastructure_bboxes, infrastructure_classes = [], []
        objects_bboxes, objects_classes = self.cluster_objects(sem_scan)
        agents_bboxes, agents_classes, agents_instances = self.cluster_agents(sem_scan)

        self.init_nodes(
            terrain_classes, objects_classes, objects_bboxes,
            agents_classes, agents_bboxes, agents_instances,
            infrastructure_classes, infrastructure_bboxes
        )
        self.add_edges(sem_scan)
        return self.ssg

    ## Initialize the SSG nodes.
    def init_nodes(
        self, terrain_classes:list, objects_classes:list, objects_bboxes:list,
        agents_classes:list, agents_bboxes:list, agents_instances:list,
        infrastructure_classes:list, infrastructure_bboxes:list
    ) -> None:
        self.ssg.add_node(0, layer=0, semantic_id=0, instance_id=-1,
                          center=[0, 0, 0], extent=[0, 0, 0], orientation=np.eye(3).tolist(),
                          num_points=0, points=None, latent=None)
        for i, class_id in enumerate(terrain_classes, start=self.ssg.number_of_nodes()):
            num_points = np.sum(self.sem_scan.sem_label == class_id)
            self.ssg.add_node(i, layer=1, semantic_id=class_id, instance_id=-1,
                              center=[0, 0, 0], extent=[0, 0, 0], orientation=np.eye(3).tolist(),
                              num_points=num_points, points=None, latent=None)
        self.ssg.add_node(self.ssg.number_of_nodes(), layer=1, semantic_id=49,
                          instance_id=-1, center=[0,0,0], extent=[0,0,0], orientation=np.eye(3).tolist(),
                          num_points=0, points=None, latent=None)
        self.ssg.graph['per_layer']['terrain'] = len(terrain_classes)+1
        for i, class_id, obb in zip(
            range(self.ssg.number_of_nodes(), 
                  self.ssg.number_of_nodes()+len(objects_classes)), 
            objects_classes, objects_bboxes
        ):
            class_mask = self.sem_scan.sem_label == class_id
            points_in_class = self.sem_scan.points[class_mask]
            min_bound = obb.get_min_bound()
            max_bound = obb.get_max_bound()
            bbox_mask = np.all((points_in_class >= min_bound) & (points_in_class <= max_bound), axis=1)
            num_points = np.sum(bbox_mask)
            self.ssg.add_node(
                i, layer=2, semantic_id=class_id, instance_id=-1,
                center=obb.center.tolist(), extent=obb.extent.tolist(), orientation=obb.R.tolist(),
                num_points=num_points, points=None, latent=None
            )
        self.ssg.graph['per_layer']['objects'] = len(objects_classes)
        for i, class_id, instance_id, obb in zip(
            range(self.ssg.number_of_nodes(), 
                  self.ssg.number_of_nodes()+len(agents_classes)), 
            agents_classes, agents_instances, agents_bboxes
        ):
            class_mask = self.sem_scan.sem_label == class_id
            points_in_class = self.sem_scan.points[class_mask]
            min_bound = obb.get_min_bound()
            max_bound = obb.get_max_bound()
            bbox_mask = np.all((points_in_class >= min_bound) & (points_in_class <= max_bound), axis=1)
            num_points = np.sum(bbox_mask)
            self.ssg.add_node(
                i, layer=3, semantic_id=class_id, instance_id=instance_id,
                center=obb.center.tolist(), extent=obb.extent.tolist(), orientation=obb.R.tolist(),
                num_points=num_points, points=None, latent=None
            )
        self.ssg.graph['per_layer']['agents'] = len(agents_classes)
        if self.layer_4:
            for i, class_id, obb in zip(
                range(self.ssg.number_of_nodes(),
                      self.ssg.number_of_nodes()+len(infrastructure_classes)), 
                infrastructure_classes, infrastructure_bboxes
            ):
                class_mask = self.sem_scan.sem_label == class_id
                points_in_class = self.sem_scan.points[class_mask]
                min_bound = obb.get_min_bound()
                max_bound = obb.get_max_bound()
                bbox_mask = np.all((points_in_class >= min_bound) & (points_in_class <= max_bound), axis=1)
                num_points = np.sum(bbox_mask)
                self.ssg.add_node(
                    i, layer=4, semantic_id=class_id, instance_id=-1,
                    center=obb.center.tolist(), extent=obb.extent.tolist(), orientation=obb.R.tolist(),
                    num_points=num_points, points=None, latent=None
                )
            self.ssg.graph['per_layer']['infrastructure'] = len(infrastructure_classes)
        if self.verbose:
            print(f'Graph attributes: {self.ssg.graph}')        
            print(f'SSG: {self.ssg.nodes(data=True)}')
        return
    
    ## Add edges to the SSG.
    def add_edges(self, sem_scan:SemLaserScan) -> None:
        for node in self.ssg.nodes(data=True):
            if node[1]['layer'] == 1:
                self.ssg.add_edge(0, node[0])
        if self.layer_4:
            other_ground_node = [n[0] for n in self.ssg.nodes(data=True) if n[1]['semantic_id'] == 49][0]
            for node in self.ssg.nodes(data=True):
                if node[1]['layer'] == 4:
                    self.ssg.add_edge(other_ground_node, node[0])
        terrain_node_ids = [n[0] for n in self.ssg.nodes(data=True) if n[1]['layer'] == 1]
        object_nodes = [(n[0], n[1]) for n in self.ssg.nodes(data=True) if n[1]['layer'] == 2]
        agent_nodes = [(n[0], n[1]) for n in self.ssg.nodes(data=True) if n[1]['layer'] == 3]
        self.find_connections(object_nodes, terrain_node_ids, sem_scan)
        self.find_connections(agent_nodes, terrain_node_ids, sem_scan)
        return

    ## Find connections between nodes.
    def find_connections(self, source_nodes:list, target_node_ids:list, sem_scan:SemLaserScan) -> None:
        for source_id, source_data in source_nodes:
            source_center = source_data['center']
            min_dist = float('inf')
            closest_target = None
            for target_id in target_node_ids:
                target_class = self.ssg.nodes[target_id]['semantic_id']
                if target_class in self.vocabulary['terrain_classes']:
                    target_points = sem_scan.points[sem_scan.sem_label == target_class]
                    if len(target_points) > 0:
                        dist = np.min(np.linalg.norm(target_points - source_center, axis=1))
                        if dist < min_dist:
                            min_dist = dist
                            closest_target = target_id
            if closest_target is not None:
                self.ssg.add_edge(source_id, closest_target)
        return

    ## Get terrain classes.
    def get_terrain_classes(self, sem_scan:SemLaserScan) -> list:
        terrain_mask = np.isin(sem_scan.sem_label, self.vocabulary['terrain_classes'])
        terrain_labels = sem_scan.sem_label[terrain_mask]
        return np.unique(terrain_labels)
        
    ## Clustering terrain.
    def cluster_terrain(self, sem_scan:SemLaserScan) -> None:
        pass

    # ## Clustering infrastructure.
    # def cluster_infrastructure(self, sem_scan:SemLaserScan) -> tuple[list, list]:
    #     bboxes, classes = [], []
    #     # Parameters for DBSCAN clustering.
    #     eps = 1.5   # Maximum distance for points to be considered in the same neighborhood.
    #     min_points = 8  # Minimum number of points to form a cluster.
    #     tic = time.time()
    #     for infrastructure_class in self.vocabulary['infrastructure_classes']:
    #         # Extract points for the current infrastructure class.
    #         infrastructure_points = sem_scan.points[sem_scan.sem_label == infrastructure_class]
    #         if len(infrastructure_points) == 0:
    #             continue

    #         # Create an Open3D point cloud.
    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(infrastructure_points)
            
    #         # Use Open3D's DBSCAN clustering (region growing style).
    #         toc = time.time()
    #         labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    #         unique_labels = np.unique(labels)
            
    #         for cluster_label in unique_labels:
    #             if cluster_label < 0:  # Skip noise.
    #                 continue
    #             indices = np.where(labels == cluster_label)[0]
    #             if len(indices) < min_points:
    #                 continue
    #             cluster_points = np.asarray(pcd.points)[indices, :]
    #             obb = o3d.geometry.OrientedBoundingBox.create_from_points(
    #                 o3d.utility.Vector3dVector(cluster_points)
    #             )
    #             bboxes.append(obb)
    #             classes.append(infrastructure_class)
    #     return bboxes, classes
    
    

    # #     # Old (plane fitting) implementation kept for reference:
    # def cluster_infrastructure(self, sem_scan:SemLaserScan) -> tuple[list, list]:
    #     bboxes, classes = [], []
    #     for infrastructure_class in self.vocabulary['infrastructure_classes']:
    #         infrastructure_points = sem_scan.points[sem_scan.sem_label == infrastructure_class]
    #         if len(infrastructure_points) > 0:
    #             pcd = o3d.geometry.PointCloud()
    #             pcd.points = o3d.utility.Vector3dVector(infrastructure_points)
    #             voxel_size = 0.001 #0.005 # 0.075
    #             pcd = pcd.voxel_down_sample(voxel_size)
    #             pcd.estimate_normals(
    #                 search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=50) # 75
    #             )
    #             oboxes = pcd.detect_planar_patches(
    #                 normal_variance_threshold_deg=60, # 30, 60
    #                 coplanarity_deg=60, # 30, 60
    #                 outlier_ratio=0.1, # 0.075
    #                 min_plane_edge_length=1.5, # 1.5, 2.0
    #                 min_num_points=16, # 20
    #                 search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50)
    #             )
    #             for obox in oboxes:
    #                 oriented_bbox = obox.get_oriented_bounding_box()
    #                 bboxes.append(oriented_bbox)
    #                 classes.append(infrastructure_class)
    #     return bboxes, classes

    def cluster_infrastructure(self, sem_scan: SemLaserScan) -> tuple:
        bboxes, classes = [], []
        for infrastructure_class in self.vocabulary['infrastructure_classes']:
            # Extract points for the current infrastructure class.
            infrastructure_points = sem_scan.points[sem_scan.sem_label == infrastructure_class]
            if len(infrastructure_points) == 0:
                continue
            # Use plane detection for buildings and fences (classes 50 and 51)
            if infrastructure_class in [50, 51]:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(infrastructure_points)
                voxel_size = 0.001  # adjust voxel size as needed
                pcd = pcd.voxel_down_sample(voxel_size)
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=50)
                )
                oboxes = pcd.detect_planar_patches(
                    normal_variance_threshold_deg=60,
                    coplanarity_deg=60,
                    outlier_ratio=0.1,
                    min_plane_edge_length=1.5,
                    min_num_points=16,
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50)
                )
                for obox in oboxes:
                    oriented_bbox = obox.get_oriented_bounding_box()
                    bboxes.append(oriented_bbox)
                    classes.append(infrastructure_class)
            # Use DBSCAN region growing for vegetation (class 70)
            elif infrastructure_class == 70:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(infrastructure_points)
                eps = 0.5   # Maximum distance for points to be considered neighbors
                min_points = 32  # Minimum number of points to form a cluster
                labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
                unique_labels = np.unique(labels)
                for cluster_label in unique_labels:
                    if cluster_label < 0:  # Skip noise.
                        continue
                    indices = np.where(labels == cluster_label)[0]
                    if len(indices) < min_points:
                        continue
                    cluster_points = np.asarray(pcd.points)[indices, :]
                    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                        o3d.utility.Vector3dVector(cluster_points)
                    )
                    bboxes.append(obb)
                    classes.append(infrastructure_class)
        return bboxes, classes


    ## Clustering objects.
    def cluster_objects(self, sem_scan:SemLaserScan) -> tuple:
        bboxes, classes = [],[]
        for object_class in self.vocabulary['object_classes']:
            object_points = sem_scan.points[sem_scan.sem_label == object_class]
            if len(object_points) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(object_points)
                labels = np.array(pcd.cluster_dbscan(eps=0.75, min_points=10, print_progress=False))
                if len(labels) > 0:
                    unique_labels = np.unique(labels[labels >= 0])
                    for cluster_id in unique_labels:
                        cluster_points = object_points[labels == cluster_id]
                        cluster_pcd = o3d.geometry.PointCloud()
                        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                        oriented_bbox = cluster_pcd.get_oriented_bounding_box()
                        bboxes.append(oriented_bbox)
                        classes.append(object_class)
        return bboxes, classes

    ## Clustering agents.
    def cluster_agents(self, sem_scan:SemLaserScan) -> tuple:
        bboxes, classes, instances = [], [], []
        for agent_class in self.vocabulary['agent_classes']:
            agent_mask = sem_scan.sem_label == agent_class
            agent_points = sem_scan.points[agent_mask]
            agent_instances = sem_scan.inst_label[agent_mask]
            if len(agent_points) > 0:
                unique_instances = np.unique(agent_instances[agent_instances > 0])
                for instance_id in unique_instances:
                    instance_points = agent_points[agent_instances == instance_id]
                    if len(instance_points) >= 4:  # Ensure there are at least 4 points
                        instance_pcd = o3d.geometry.PointCloud()
                        instance_pcd.points = o3d.utility.Vector3dVector(instance_points)
                        oriented_bbox = instance_pcd.get_oriented_bounding_box()
                        bboxes.append(oriented_bbox)
                        classes.append(agent_class)
                        instances.append(instance_id)
        return bboxes, classes, instances

    ## Populate SSG points from patches.
    # def populate_patches(self, patches: list, latent:list=None, populate_points:bool=True) -> None:
    #     """
    #     Populates SSG node fields ('points') from a list of patches extracted separately.
    #     """
    #     for idx, patch in enumerate(patches):
    #         for node_id, data in self.ssg.nodes(data=True):
    #             ## Handle the case for the layer 1 node (terrain) where all patches of the same class are merged.
    #             if data['layer'] == 1 and data['semantic_id'] == patch['semantic_id']:
    #                 if populate_points:
    #                     try:
    #                         self.ssg.nodes[node_id]['points'] = np.concatenate((self.ssg.nodes[node_id]['points'], patch['points']), axis=0)
    #                         ## Replace num_points with the number of points in the patch
    #                         # self.ssg.nodes[node_id]['num_points'.] = len(self.ssg.nodes[node_id]['points'])
    #                         # print(f"Populating node {node_id} with {len(self.ssg.nodes[node_id]['points'])} points")
    #                     except:
    #                         self.ssg.nodes[node_id]['points'] = patch['points']
    #                         ## Replace num_points with the number of points in the patch
    #                         # self.ssg.nodes[node_id]['num_points'] = len(patch['points'])
    #                         # print(f"Populating node {node_id} with {len(self.ssg.nodes[node_id]['points'])} points")
    #                 ## For latent, we can keep a list of latent vectors for each node.
    #                 if latent is not None:
    #                     if self.ssg.nodes[node_id]['latent'] is None:
    #                         self.ssg.nodes[node_id]['latent'] = []
    #                     self.ssg.nodes[node_id]['latent'].append(latent[idx])
    #                     # print(f"Populating node {node_id} with latent vector of shape {latent[idx].shape}")
    #             ## For other nodes, we check if the semantic_id and layer match.
    #             elif (data['semantic_id'] == patch['semantic_id'] and data['layer'] == patch['layer']
    #                 and np.allclose(data['center'], patch['center'], atol=1e-3)):
    #                 if populate_points:
    #                     self.ssg.nodes[node_id]['points'] = patch['points']            
    #                     ## Replace num_points with the number of points in the patch
    #                     # self.ssg.nodes[node_id]['num_points'] = len(patch['points'])
    #                     # print(f"Populating node {node_id} with {len(patch['points'])} points")
    #                 if latent is not None:
    #                     self.ssg.nodes[node_id]['latent'] = latent[idx]
    #                     # print(f"Populating node {node_id} with latent vector of shape {latent[idx].shape}")
    #                 break
    #     return

    def populate_patches(self, patches: list, latent: list = None, populate_points: bool = True) -> None:
        """
        Populates SSG node fields ('points', 'latent', 'center', 'extent', 'orientation') from a list of patches.
        For terrain (layer 1) nodes, these fields are lists—one per patch in that node.
        """
        for idx, patch in enumerate(patches):
            for node_id, data in self.ssg.nodes(data=True):
                # --- Terrain nodes: merge patches and keep lists ---
                if data['layer'] == 1 and data['semantic_id'] == patch['semantic_id']:
                    # Points: concatenate as before
                    if populate_points:
                        try:
                            self.ssg.nodes[node_id]['points'] = np.concatenate(
                                (self.ssg.nodes[node_id]['points'], patch['points']), axis=0)
                        except Exception:
                            self.ssg.nodes[node_id]['points'] = patch['points']
                    # Latents: append to list
                    if latent is not None:
                        if self.ssg.nodes[node_id].get('latent', None) is None:
                            self.ssg.nodes[node_id]['latent'] = []
                        self.ssg.nodes[node_id]['latent'].append(latent[idx])
                    # Centers, extents, orientations: append to lists in their respective fields
                    for attr in ['center', 'extent', 'orientation']:
                        val = patch[attr]
                        # Initialize as list if not already (for terrain nodes)
                        if not isinstance(self.ssg.nodes[node_id][attr], list):
                            self.ssg.nodes[node_id][attr] = []
                        self.ssg.nodes[node_id][attr].append(val)
                    break

                # --- Other nodes: update as before ---
                elif (data['semantic_id'] == patch['semantic_id'] and data['layer'] == patch['layer']
                    and np.allclose(data['center'], patch['center'], atol=1e-3)):
                    if populate_points:
                        self.ssg.nodes[node_id]['points'] = patch['points']
                    if latent is not None:
                        self.ssg.nodes[node_id]['latent'] = latent[idx]
                    for attr in ['center', 'extent', 'orientation']:
                        self.ssg.nodes[node_id][attr] = patch[attr]
                    break
        return

    ## Visualize the SSG
    def visualize_ssg(self, ssg:nx.Graph, color_map:dict) -> None:
        """ Visualize the SSG in 3D based on node centers """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Remove background color and axis
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        ax.set_axis_off()

        # Create position dictionary based on node centers
        pos = {}
        for node, data in ssg.nodes(data=True):
            if data['layer'] == 1:
                if data['semantic_id'] == 40:
                    center = [0, 0, 0]
                elif data['semantic_id'] == 44:
                    center = [15, 0, 0]
                elif data['semantic_id'] == 48:
                    center = [-15, 0, 0]
                elif data['semantic_id'] == 49:
                    center = [0, -10, 0]
                elif data['semantic_id'] == 72:
                    center = [0, 10, 0]
                pos[node] = center
            else:
                pos[node] = data['center'].copy()
            # pos[node][2] = -data['layer'] * 5  # Offset z based on layer            
            if data['layer'] == 0:
                pos[node][2] = 0
            elif data['layer'] == 1:
                pos[node][2] = -5
            elif data['layer'] == 2:
                pos[node][2] = -15
            elif data['layer'] == 3:
                pos[node][2] = -20
            elif data['layer'] == 4:
                pos[node][2] = -10

        # Get semantic IDs and convert kitti colors from BGR to RGB
        semantic_ids = nx.get_node_attributes(ssg, 'semantic_id')
        colors = []
        for node in ssg.nodes():
            sem_id = semantic_ids[node]
            # Get BGR color from kitti config and convert to RGB
            bgr_color = color_map[sem_id]
            rgb_color = [c/255.0 for c in bgr_color[::-1]]  # Reverse BGR to RGB and normalize to [0,1]
            colors.append(rgb_color)

        # Plot nodes by layer with different markers
        layer_markers = {
            0: 'o',  # Circle for root
            1: 's',  # Square for terrain
            2: '^',  # Triangle up for objects
            3: 'o',  # Diamond for agents
            4: 'v'   # Triangle down for infrastructure
        }

        # Plot each layer separately to use different markers
        for layer in range(5):
            if layer_nodes := [
                n for n in ssg.nodes() if ssg.nodes[n]['layer'] == layer
            ]:
                xs = [pos[node][0] for node in layer_nodes]
                ys = [pos[node][1] for node in layer_nodes]
                zs = [pos[node][2] for node in layer_nodes]
                node_colors = [colors[list(ssg.nodes()).index(n)] for n in layer_nodes]
                ax.scatter(xs, ys, zs, c=node_colors, s=500, alpha=0.75, marker=layer_markers[layer])
        # Draw edges with colors based on layer connections
        layer_colors = {
            (0,1): 'black',  # Connections between layer 0 and 1
            (1,2): 'green',  # Connections between layer 1 and 2
            (1,3): 'red',    # Connections between layer 1 and 3
            (1,4): 'blue',   # Connections between layer 1 and 4
        }

        for edge in ssg.edges():
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            z = [pos[edge[0]][2], pos[edge[1]][2]]

            # Get layers of connected nodes
            layer1 = ssg.nodes[edge[0]]['layer']
            layer2 = ssg.nodes[edge[1]]['layer']
            # Sort layers to match dictionary keys
            layer_pair = tuple(sorted([layer1, layer2]))

            edge_color = layer_colors.get(layer_pair, 'gray')  # Default to gray if not in dictionary
            ax.plot(x, y, z, c=edge_color, alpha=0.5, linewidth=1.5)

        # Add labels with instance IDs as text
        for node, (x, y, z) in pos.items():
            # instance_id = ssg.nodes[node]['instance_id']
            semantic_id = ssg.nodes[node]['semantic_id']
            ax.text(x, y, z, str(semantic_id), fontsize=12, color='black', fontweight='bold',
                    horizontalalignment='center', verticalalignment='center')

        ## Set axis limits
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-20, 0)
        plt.tight_layout()
        plt.title("3D Semantic Scene Graph")
        plt.show()
        # save the plot
        return

    def visualize_scan_and_bboxes(self, sem_scan: SemLaserScan, color_map) -> None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sem_scan.points)
        bgr_color = np.array([color_map[sem_id] for sem_id in sem_scan.sem_label])
        rgb_color = bgr_color[:, ::-1] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb_color)
        bboxes = []
        for node in self.ssg.nodes(data=True):
            if node[1]['layer'] > 0:
                c = np.array(node[1]['center'])
                e = np.array(node[1]['extent'])
                R = np.array(node[1]['orientation'])
                obb = o3d.geometry.OrientedBoundingBox(c, R, e)
                obb.color = [0, 0, 0]
                bboxes.append(obb)
        o3d.visualization.draw_geometries([pcd] + bboxes)
        return
