import os
import torch
import trimesh
import numpy as np
from argparse import ArgumentParser
from scene import Scene, MeshGaussianModel
from scipy.spatial.transform import Rotation as R

class Translate:
    def __init__(self, args, model_paths: str, source_path: str = '', iterations: int = 30000):
        self.args = args
        self.model_paths = model_paths
        self.source_path = source_path
        self.iterations = iterations
        self.sh_degree = args.sh_degree
        self.gaussians = None
        self.ply_path = ""
        self.is_training = False
        self.source_mesh_path = args.source_mesh
        self.target_mesh_path = args.target_mesh
        self._init_models(iterations)
        
        # Compute transformations
        self.rotation_matrices, self.translation_vectors = self.compute_transformations()

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
            source_mesh = trimesh.load(self.source_mesh_path)
            deformed_mesh = trimesh.load(self.target_mesh_path)
            self.gaussians = MeshGaussianModel(source_mesh, sh_degree=self.sh_degree)
            self.gaussians.load_ply(self.ply_path)
            self.gaussians.load_mesh(source_mesh)
            self.gaussians.update_binding()

            move_gaussian(self.gaussians)

            # Save self.gaussian.binding as an npz file
            binding_file_path = os.path.join(self.model_paths, "gaussian_binding.npz")
            np.savez(binding_file_path, binding=self.gaussians.binding.cpu().numpy())
            print(f'[INFO] Gaussian binding saved to: {binding_file_path}')

    def compute_transformations(self):
        source_mesh = trimesh.load(self.source_mesh_path)
        target_mesh = trimesh.load(self.target_mesh_path)
        
        source_faces = source_mesh.triangles
        target_faces = target_mesh.triangles
        
        num_faces = len(source_faces)
        rotation_matrices = []
        translation_vectors = []
        
        for i in range(num_faces):
            src_face = source_faces[i]
            tgt_face = target_faces[i]
            
            # Calculate centroids
            src_centroid = np.mean(src_face, axis=0)
            tgt_centroid = np.mean(tgt_face, axis=0)
            
            # Subtract centroids
            src_face_centered = src_face - src_centroid
            tgt_face_centered = tgt_face - tgt_centroid
            
            # Compute covariance matrix
            H = src_face_centered.T @ tgt_face_centered
            
            # SVD for the rotation
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure a right-handed coordinate system
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Translation vector
            t = tgt_centroid - R @ src_centroid
            
            rotation_matrices.append(R)
            translation_vectors.append(t)
        
        return np.array(rotation_matrices), np.array(translation_vectors)

    def move_gaussian(self, gaussian):
        binding_mat = gaussian.binding  # Gaussian to face index 
        location = gaussian.get_xyz()
        rotation = gaussian.get_rotation()

        # Apply new location and rotation property based on rotation translation and gaussian binding properties
        valid_indices = (binding_mat >= 0) & (binding_mat < len(self.rotation_matrices))
        
        if np.any(valid_indices):
            valid_binding = binding_mat[valid_indices]

            R_matrices = self.rotation_matrices[valid_binding]
            t_vectors = self.translation_vectors[valid_binding]

            new_locations = np.einsum('ijk,ik->ij', R_matrices, location[valid_indices]) + t_vectors
            new_rotations = np.einsum('ijk,ik->ij', R_matrices, rotation[valid_indices])

            # Assign the new values back to the valid indices
            location[valid_indices] = new_locations
            rotation[valid_indices] = new_rotations

        gaussian._xyz = location
        gaussian._rotation = rotation


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_paths", type=str, nargs="+")
    parser.add_argument("--source_path", "-s", type=str, default="")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonics degree")
    parser.add_argument("--float32_matmul_precision", type=str, default=None, help="Set torch float32 matmul precision")
    parser.add_argument('--source_mesh', type=str, required=True, help='Path to the source mesh file')  # No shorthand
    parser.add_argument('--target_mesh', type=str, required=True, help='Path to the target mesh file')  # No shorthand

    args, unknown_args = parser.parse_known_args()

    # Set torch float32 matmul precision if provided
    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
        del args.float32_matmul_precision

    # Initialize the Translate with provided arguments
    translate = Translate(args, model_paths=args.model_paths[0], source_path=args.source_path, iterations=30000)

    print("\nProcess complete.")
    print("Rotation Matrices:")
    for R in translate.rotation_matrices:
        print(R)
    print("Translation Vectors:")
    for t in translate.translation_vectors:
        print(t)
