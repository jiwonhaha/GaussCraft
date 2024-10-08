## 🚀 **GaussCraft: Editable 2D Gaussian Splatting with Mesh Deformation**

Explore the project:

- [**GitHub Repository**](https://github.com/jiwonhaha/cgvi_thesis) 💻
- [**Project Page**](https://jiwonhaha.github.io/gausscraft/) 🌐


## Abstract:

2D Gaussian Splatting has recently provided a novel method for accurately recon- structing geometrically consistent radiance fields from multi-view images, improv- ing surface representation and achieving high-quality, real-time rendering. However, existing 2D or 3D Gaussian splatting methods do not offer the capability for user- directed scene editing. While some 3D Gaussian-based methods exist for avatar specific editing, they are limited to avatar applications and do not extend to general scenarios. Therefore, this paper introduces GaussCraft, an real-time scene editing framework that utilizes 2D Gaussian Splatting for high-quality mesh reconstruction. Unlike previous methods, GaussCraft does not require retraining the model for each edited scene; it is trained once using the reconstructed mesh. Specifically, Gauss- Craft reconstructs the mesh from multi-view images using 2D Gaussian Splatting, and then binds 2D Gaussians to each mesh face, allowing users to render scenes with user-edited, deformed meshes. This method has been tested on both synthetic and real-world captured data, showing significant potential for application across various fields. 

## Model Overview
![Model Overview](figure/2dgs_edit_main.png)

### Result Images

<div style="display: flex;">
    <div style="flex: 1; padding: 5px;">
        <p>Evaluation on NeRF Synthetic Dataset</p>
        <img src="figure/nerf_eval.png" alt="NeRF Evaluation" style="width: 100%;">
    </div>
</div>

## Installation

### Using an Existing Conda Environment

If you already have a conda environment set up for 2D Gaussian Splatting, you can use it directly. Otherwise, follow the instructions below for a new installation.

### New Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/jiwonhaha/GaussCraft.git --recursive
    cd GaussCraft
    ```

2. Create and activate the conda environment:
    ```bash
    conda env create --file environment.yml
    conda activate surfel_splatting
    ```

3. Install the required Python packages for viewer:
    ```bash
    pip install viser==0.1.29
    pip install splines
    pip install lightning
    ```

## Training

### Train 2D Gaussian Splatting to Reconstruct Mesh and Deform It

(*Alternatively, users can skip this process and use lego from NeRF Synthetic Dataset with pre-provided meshes and deformed meshes from [Google Drive](https://drive.google.com/drive/folders/1-_z_Ojb2abcQAQVpitrGvGyLjbctUfvI?usp=share_link).*)

1. Use the original 2D GS code to start:
    ```bash
    python train.py -s <data source path> -m <output data path>
    ```

2. Extract the mesh:
    ```bash
    python render.py -s <data source path> -m <output data path> --depth_ratio 1 --skip_test --skip_train
    ```

3. Using ARAP Mesh Deformation:
    User can use interactive selection for handles and static vertices:
    ```bash
    python arap.py <path to reconstructed mesh> 
    ```

### Bind Gaussian to Extracted mesh and Render with Deformed mesh

1. Train 2D GS with binding to the mesh:
    ```bash
    python train.py -s <data source path> -m <output data path> --mesh_path <path to original mesh>
    ```

2. View the edited rendering with viewer:
    ```bash
    python viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path> --mesh_path <path to deformed mesh>
    ```


## Testing

```bash
python matric_mesh.py --gt_dir <path to gt dir> --test_dir <path to test dir>
```

### Bounded Mesh Extraction

To export a mesh within a bounded volume, simply use:
```bash
python render.py -m <path to pre-trained model> -s <path to dataset> 
```
Commandline arguments you should adjust accordingly for meshing for bounded TSDF fusion, use
```bash
--depth_ratio # 0 for mean depth and 1 for median depth
--voxel_size # voxel size
--depth_trunc # depth truncation
```
If these arguments are not specified, the script will automatically estimate them using the camera information.
### Unbounded Mesh Extraction
To export a mesh with an arbitrary size, we devised an unbounded TSDF fusion with space contraction and adaptive truncation.
```bash
python render.py -m <path to pre-trained model> -s <path to dataset> --mesh_res 1024
```

**Custom Dataset**: GaussCraft uses the same COLMAP loader as 3DGS, you can prepare your data as 3DGS [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes). 


### Acknowledgements

This project is built upon [2DGS](https://github.com/hbb1/2d-gaussian-splatting) and [Gaussian Avatar](https://github.com/ShenhanQian/GaussianAvatars). 
