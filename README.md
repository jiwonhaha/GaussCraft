# Simple Viser Viewer for 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

[Editable 2D GS](https://github.com/jiwonhaha/cgvi_thesis)

This repository provides an editable 2D Gaussian splatting implementation using ARAP mesh deformation.

## Installation

### Using an Existing Conda Environment

If you already have the conda environment for 2D GS, you can use it directly. Otherwise, follow the installation instructions below:

### New Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/jiwonhaha/cgvi_thesis.git --recursive
    cd cgvi_thesis
    ```

2. Create and activate the conda environment:
    ```bash
    conda env create --file environment.yml
    conda activate surfel_splatting
    ```

3. Install the required Python packages:
    ```bash
    pip install viser==0.1.29
    pip install splines
    pip install lightning
    ```

## Usage

### Viewing a 2D GS PLY File

1. Use the original 2D GS code to start:
    ```bash
    python train_mesh.py -s <data source path> -m <output data path>
    ```

2. Render the 2D GS:
    ```bash
    python render.py -s <data source path> -m <output data path> --depth_ratio 1 --skip_test --skip_train
    ```

### Extracting and Training the Mesh

1. Extract the mesh:
    ```bash
    python train_mesh.py -s <data source path> -m <output data path> --mesh_path <path to original mesh>
    ```

2. Train 2D GS with binding to the mesh:
    ```bash
    python arap.py <path to original mesh> handle_vertex_index move_scale how_many_static_vertices_around_static_vertex given_static_vertex_index
    ```

### Using ARAP Mesh Deformation

1. View the deformed rendering:
    ```bash
    python viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path> --mesh_path <path to deformed mesh>
    ```
