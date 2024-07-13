import os
import torch
import trimesh
import numpy as np
from argparse import ArgumentParser
from scene import Scene, MeshGaussianModel

class Allocate:
    def __init__(self, args, model_paths: str, source_path: str = '', iterations: int = 30000):
        self.args = args
        self.model_paths = model_paths
        self.source_path = source_path
        self.iterations = iterations
        self.sh_degree = 3 
        self.gaussians = None
        self.ply_path = ""
        self.is_training = False
        self.mesh_path = args.mesh  # Ensure mesh path is included in the instance
        self._init_models(iterations)

    def _init_models(self, iterations):
        # init gaussian model & renderer
        if not self.is_training:
            self.iteration = iterations
            if not self.model_paths.lower().endswith('.ply'):
                self.ply_path = os.path.join(self.model_paths, "point_cloud", f"iteration_{iterations}", "point_cloud.ply")
            else:
                self.ply_path = self.model_paths

            if not os.path.exists(self.ply_path):
                print(self.ply_path)
                raise FileNotFoundError

            print('[INFO] ply path loaded from:', self.ply_path)
            mesh = trimesh.load(self.mesh_path)
            self.gaussians = MeshGaussianModel(mesh, sh_degree=self.sh_degree)
            self.gaussians.load_ply(self.ply_path)
            self.gaussians.load_mesh(mesh)
            self.gaussians.update_binding()

            # Save self.gaussian.binding as an npz file
            binding_file_path = os.path.join(self.model_paths, "gaussian_binding.npz")
            np.savez(binding_file_path, binding=self.gaussians.binding.cpu())
            print(f'[INFO] Gaussian binding saved to: {binding_file_path}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_paths", type=str, nargs="+")
    parser.add_argument("--source_path", "-s", type=str, default="")
    parser.add_argument("--sh_degree", type=int, default = 3, help="Spherical harmonics degree")
    parser.add_argument("--float32_matmul_precision", type=str, default=None, help="Set torch float32 matmul precision")
    parser.add_argument('--mesh', type=str, required=True, help='Path to the mesh file')  # No shorthand

    args, unknown_args = parser.parse_known_args()

    # Set torch float32 matmul precision if provided
    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
        del args.float32_matmul_precision

    # Initialize the Allocate with provided arguments
    allocate = Allocate(args, model_paths=args.model_paths[0], source_path=args.source_path, iterations=30000)

    print("\nProcess complete.")
