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
        self.binding = None #splattings to faces

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
        
    def update_binding(self):
        # Ensure vertices and faces are available
        if self.faces == None or self.verts == None:
            raise ValueError("Mesh faces and vertices must be defined")

        # Ensure the vertices and faces tensors are on the same device
        faces = self.faces.cuda()
        verts = self.verts.cuda()

        # Compute the distance from each point to each face centroid
        point_expanded = self.get_xyz.unsqueeze(1)  # Shape: (num_points, 1, 3)
        centroid_expanded = self.face_center.unsqueeze(0)  # Shape: (1, num_faces, 3)

        # Calculate distances
        distances = torch.norm(point_expanded - centroid_expanded, dim=2)  # Shape: (num_points, num_faces)

        # Find the index of the closest face for each point
        closest_face_indices = torch.argmin(distances, dim=1)

        # Update binding
        self.binding = closest_face_indices

    def capture(self):
        """Capture the current state of the Gaussian model, including binding."""
        state = super().capture()
        state['verts'] = self.verts
        state['faces'] = self.faces
        state['binding'] = self.binding
        return state

    def restore(self, state, opt):
        """Restore the Gaussian model from the given state, including binding."""
        super().restore(state, opt)
        self.verts = state['verts']
        self.faces = state['faces']
        self.binding = state['binding']


