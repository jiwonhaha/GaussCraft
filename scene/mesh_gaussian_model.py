import torch
from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
import numpy as np

class MeshGaussianModel(GaussianModel):
    def __init__(self, mesh, deformed_mesh, sh_degree: int):
        super().__init__(sh_degree)
        self.verts = None
        self.faces = None 
        self.binding = None #splattings to faces
        self.binding_counter = None

        self.face_center = None
        self.face_scaling = None
        self.face_orien_mat = None
        self.face_orien_quat = None

        #check amount of shape change for deformation
        self.mesh_diff = None
        self.diff_threshold = None

        # binding is initialized once the mesh topology is known
        if mesh is not None:
            self.load_mesh(mesh, deformed_mesh)
            self.binding = torch.arange(len(self.faces)).cuda()
            self.binding_counter = torch.ones(len(self.faces), dtype=torch.int32).cuda()
            


    def load_mesh(self, mesh, deformed_mesh):
        """Load a single general triangular mesh."""
        self.verts = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
        self.faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()


        self.update_mesh_properties()
        if deformed_mesh is not None:
            self.cal_mesh_diff(mesh, deformed_mesh)

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

    def cal_mesh_diff(self, mesh, deformed_mesh):
        """Calculate and compare the intensity of the shape difference between the original mesh and the deformed mesh."""
        # Load vertices and faces from both original and deformed meshes
        original_verts = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
        deformed_verts = torch.tensor(deformed_mesh.vertices, dtype=torch.float32).cuda()

        # Compute face orientation and scale differences
        original_orien_mat, original_scale = compute_face_orientation(original_verts, torch.tensor(mesh.faces, dtype=torch.int32).cuda(), return_scale=True)
        deformed_orien_mat, deformed_scale = compute_face_orientation(deformed_verts, torch.tensor(mesh.faces, dtype=torch.int32).cuda(), return_scale=True)
        
        orien_diff = torch.norm(
            quat_xyzw_to_wxyz(rotmat_to_unitquat(original_orien_mat)) - 
            quat_xyzw_to_wxyz(rotmat_to_unitquat(deformed_orien_mat)), dim=1
        )
        
        scale_diff = torch.abs(original_scale - deformed_scale).squeeze()

        # Normalize each difference to [0, 1]
        orien_diff = (orien_diff - orien_diff.min()) / (orien_diff.max() - orien_diff.min())
        scale_diff = (scale_diff - scale_diff.min()) / (scale_diff.max() - scale_diff.min())
        
        # Combine the normalized differences
        combined_diff = orien_diff + scale_diff

        # Normalize the combined differences to [0, 1]
        self.mesh_diff = (combined_diff - combined_diff.min()) / (combined_diff.max() - combined_diff.min())
        self.diff_threshold = torch.quantile(self.mesh_diff, 0.9)

    def remove_z_rotation_from_quaternion(self, q):
        # Assuming the quaternion q is in the format [w, x, y, z] with shape (N, 4)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Step 1: Calculate the angle of rotation around the z-axis (theta_z)
        theta_z = torch.atan2(z, torch.sqrt(w**2 + x**2 + y**2))

        # Step 2: Calculate the new components after removing the z-axis rotation
        cos_theta_z = torch.cos(theta_z)
        sin_theta_z = torch.sin(theta_z)

        # Calculate the new quaternion components
        w_new = w * cos_theta_z + z * sin_theta_z
        x_new = x * cos_theta_z + y * sin_theta_z
        y_new = y * cos_theta_z - x * sin_theta_z
        z_new = torch.zeros_like(z)  # Remove the z-axis component

        # Normalize the quaternion to ensure it's valid
        norm = torch.sqrt(w_new**2 + x_new**2 + y_new**2 + z_new**2).unsqueeze(1)

        # Stack the components back together
        new_q = torch.stack([w_new, x_new, y_new, z_new], dim=-1)

        # Normalize
        return new_q / norm

