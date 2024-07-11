import torch
from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
import numpy as np

class MeshGaussianModel(GaussianModel):
    def __init__(self, mesh, sh_degree: int):
        super().__init__(sh_degree)
        self.verts = None
        self.faces = None
        self.mesh_param = None  
        self.load_mesh(mesh)
        
        # binding is initialized once the mesh topology is known
        if self.binding is None:
      #      self.binding = torch.arange(len(self.faces)).cuda()
            self.binding_counter = torch.ones(len(self.faces), dtype=torch.int32).cuda()
        

    def load_mesh(self, mesh):
        """Load a single general triangular mesh."""
        self.verts = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
        self.faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()
        self.update_mesh_properties()

    def update_mesh_properties(self):
        """Update properties based on the loaded mesh."""
        # Calculate the center of each triangle face
        triangles = self.verts[self.faces]
        self.face_center = triangles.mean(dim=1)

        # Calculate orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(
            self.verts, self.faces, return_scale=True
        )
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))
        
    def update_binding(self, pcd):
        faces = self.faces
        verts = self.verts
    
        # Convert the point cloud to a tensor
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    
        # Compute the centroid of each face
        face_centroids = torch.mean(verts[faces], dim=1)
    
        # Compute the distance from each point to each face centroid
        point_expanded = fused_point_cloud.unsqueeze(1)  # Shape: (num_points, 1, 3)
        centroid_expanded = face_centroids.unsqueeze(0)  # Shape: (1, num_faces, 3)
    
        distances = torch.norm(point_expanded - centroid_expanded, dim=2)  # Shape: (num_points, num_faces)
    
        # Find the index of the closest face for each point
        closest_face_indices = torch.argmin(distances, dim=1)
        
        self.binding = closest_face_indices

