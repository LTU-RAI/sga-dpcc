import os, sys, time, torch, yaml, tqdm
from multiprocessing import Pool
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from modules.SSG import SSG
from sgadpcc_utils.LaserScan import SemLaserScan

class GenSSG():
    def __init__(self, ssg_config: dict, kitti_config: dict, layer_4: bool=True, verbose: bool=False):
        self.kitti_config = kitti_config
        self.ssg_config = ssg_config
        self.data_path = ssg_config['dataset']['data_path']
        self.verbose = verbose
        self.layer_4 = layer_4
        if self.verbose:
            print(f'SSG config: {self.ssg_config}')
            print(f'KITTI config: {self.kitti_config}')
        self.sem_laser_scan = SemLaserScan(sem_color_dict=self.kitti_config['color_map'], project=False)
        self.vocabulary = self.ssg_config['vocabulary']
        if self.verbose:
            print(f'Vocabulary: {self.vocabulary}')
        self.graphs = []

    def load_scan(self, filename:str, kitti_config:dict) -> SemLaserScan:
        sem_laser_scan = SemLaserScan(sem_color_dict=kitti_config['color_map'], project=False)
        
        sem_laser_scan.open_scan(filename)
        sem_laser_scan.open_label(filename.replace('velodyne', 'labels').replace('.bin', '.label'))
        sem_laser_scan.map_semantic_classes(kitti_config['learning_map'], kitti_config['learning_map_inv'])
        return sem_laser_scan

    def process_scan(self, scan_id, path_to_sequence, save, output_dir, output_ext):
        path_to_scan = os.path.join(path_to_sequence, scan_id)
        scan = self.load_scan(path_to_scan, self.kitti_config)
        ssg = SSG(vocabulary=self.vocabulary, layer_4=self.layer_4, verbose=self.verbose)
        tic = time.time()
        ssg.generate_ssg(sem_scan=scan, scan_id=scan_id.replace('.bin', ''))
        toc = time.time()
        # print(f'Time taken to generate SSG: {toc-tic:.4f}')
        if save:
            path_to_output = path_to_scan.replace('velodyne', output_dir).replace('.bin', output_ext)
            if output_ext == '.pickle':
                ssg.save_ssg(path_to_output)
            elif output_ext == '.pt':
                pyg_data = ssg.nx_to_pyg(ssg.ssg, self.kitti_config)
                torch.save(pyg_data, path_to_output)
        # print(f'Generated SSG for scan {scan_id}')
        return ssg

    def generate_dataset(self, sequence, save=False, output_dir='graph', output_ext='.pickle'):
        path_to_sequence = os.path.join(self.data_path, 'sequences', sequence, 'velodyne')
        if save and not os.path.exists(path_to_sequence.replace('velodyne', output_dir)):
            os.makedirs(path_to_sequence.replace('velodyne', output_dir))
        scan_ids = sorted(os.listdir(path_to_sequence))
        for scan_id in tqdm.tqdm(scan_ids, desc='Generating SSGs', total=len(scan_ids)):
            ssg = self.process_scan(scan_id, path_to_sequence, save, output_dir, output_ext)
        # with Pool(processes=16) as pool:
            # results = pool.starmap(self.process_scan, [(scan_id, path_to_sequence, save, output_dir, output_ext) for scan_id in scan_ids])
            # print(f'Generated {len(results)} graphs for sequence {sequence}')
        # self.graphs.extend(results)
        # return self.graphs
        return

    def generate_pickle_dataset(self, sequence, save=False):
        return self.generate_dataset(sequence, save, output_dir='graph', output_ext='.pickle')

    def generate_pyg_dataset(self, sequence: str, save: bool=False):
        return self.generate_dataset(sequence, save, output_dir='pyg', output_ext='.pt')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Semantic Scene Graphs (SSG) from SemanticKITTI scans')
    parser.add_argument('--ssg_config', type=str, default='/home/niksta/python_projects/sga-dpcc/config/config-latest.yaml', help='Path to SSG config YAML')
    args = parser.parse_args()

    with open(args.ssg_config, 'r') as f:
        ssg_config = yaml.safe_load(f)

    kitti_conf_path = ssg_config['dataset']['semantic-kitti-conf']
    with open(kitti_conf_path, 'r') as f:
        kitti_config = yaml.safe_load(f)

    ssg_gen = GenSSG(ssg_config=ssg_config, kitti_config=kitti_config)
    sequences = ['05', '06'] # ['00', '01', '02', '03', '04', '07', '08', '09', '10']
    for sequence in tqdm.tqdm(sequences, desc='Processing sequence', total=len(sequences)):
        ssg_gen.generate_pickle_dataset(sequence, save=True)
        # ssg_gen.generate_pyg_dataset(sequence, save=True)
