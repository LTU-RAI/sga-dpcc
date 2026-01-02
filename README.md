<div align="center">
  
<h1><em>Have We Scene It All?</em></h1>
<h2>Scene Graph-Aware Deep Point Cloud Compression</h2>

[![arXiv](https://img.shields.io/badge/Arxiv-2510.08512-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.08512)
[![DOI:10.1109/LRA.2025.3623045](https://img.shields.io/badge/IEEE-10.1109/LRA.2025.3623045-00629B.svg)](https://doi.org/10.1109/LRA.2025.3623045)
  <a href="https://www.youtube.com/watch?v=MFKrhtmc3Tw"><img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube" /></a>
  
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) ![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)

[**Nikolaos Stathoulopoulos**](https://github.com/nstathou) · [**Christoforos Kanellakis**](https://github.com/christoforoskanel) · [**George Nikolakopoulos**](https://github.com/geonikolak) 

</div>

<p align=center> <img src="./figures/dpcc-concept.drawio.png" width="75%" height="75%"/> </p>

<h2>💡 Introduction</h2>

**Abstract:** Efficient transmission of 3D point cloud data is critical for advanced perception in centralized and decentralized multi-agent robotic systems, especially nowadays with the growing reliance on edge and cloud-based processing. However, the large and complex nature of point clouds creates challenges under bandwidth constraints and intermittent connectivity, often degrading system performance. We propose a deep compression framework based on semantic scene graphs. The method decomposes point clouds into semantically coherent patches and encodes them into compact latent representations with semantic-aware encoders conditioned by Feature-wise Linear Modulation (FiLM). A folding-based decoder, guided by latent features and graph node attributes, enables structurally accurate reconstruction. Experiments on the SemanticKITTI and nuScenes datasets show that the framework achieves state-of-the-art compression rates, reducing data size by up to 98% while preserving both structural and semantic fidelity. In addition, it supports downstream applications such as multi-robot pose graph optimization and map merging, achieving trajectory accuracy and map alignment comparable to those obtained with raw LiDAR scans. 

<img src="./figures/original.gif" width="49%" height="50%"/> <img src="./figures/decompressed.gif" width="49%" height="50%"/>    

---

<h2> Setup <h2>

tested with python 3.8, should work with newer versions as long as you get the correct versions of pytorch torch-geometric and open3d.

In your desired working directory:

git clone https://github.com/LTU-RAI/sga-dpcc.git 

<h3> Conda environment <h3>

Using a virtual environment is not required but is recommended. You can use any virtual environment you want. A requirements list is provided in the requirements.txt

For conda:

conda create -n sga-dpcc --file requirements.txt

<h3> Dataset setup <h3>

Download and put to the desired directory the SemanticKITTI dataset as provided by the original authors https://semantic-kitti.org/dataset.html

The initial structure of the dataset is expected to have the following structure:
                path_to_dataset
                SemanticKitti
                ├── semantic-kitti.yaml
                ├── calib.txt
                |── sequences
                    ├── 00
                    │   ├── poses (contains the txt file, 4x4 transformation matrices)
                    │   │   ├── 00.txt
                    │   ├── velodyne (contains the point clouds)
                    │   │   ├── 000000.bin
                    │   |   ├── ...
                    │   ├── labels
                    │   │   ├── 000000.label
                    │   │   ├── ...
                    ├── 01
                    │   ├── ...

Later on we will create additional data for the training such as the semantic scene graphs etc. Check the ...

<h3> Config file <h3>

In the config-latest.yaml make sure to change all paths so they point to your corresponding directories for the dataset and the rest of the config files. 
The autoencoder.yaml controls the parameters for all the layers for both testing and training.
Note: in the config-latest.yaml the layer classes are defined based on the SemanticKITTI label classes. We not that the layers 3 and 4 are reversed with regard to the paper. 
---

<h2> Pre-trained weights <h2>

You can download the pretrained weights at https://drive.google.com/drive/folders/1uqyPaKKuTkqtskoj9h4XdJt-PfJAE5Mm?usp=drive_link

Add the to the weights/checkpoints/ folder.
---

<h2>📝 Citation</h2>

If you found this work useful, please cite the following publication:

```bibtex
@article{stathoulopoulos2025sgadpcc,
  author={Stathoulopoulos, Nikolaos and Kanellakis, Christoforos and Nikolakopoulos, George},
  journal={IEEE Robotics and Automation Letters}, 
  title={{Have We Scene It All? Scene Graph-Aware Deep Point Cloud Compression}}, 
  year={2025},
  volume={10},
  number={12},
  pages={12477-12484},
  doi={10.1109/LRA.2025.3623045}}
}

```
