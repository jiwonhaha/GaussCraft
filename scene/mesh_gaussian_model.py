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
        self.orien_diff = None

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
        # Original mesh vertices and faces
        original_verts = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
        original_faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()

        # Deformed mesh vertices
        deformed_verts = torch.tensor(deformed_mesh.vertices, dtype=torch.float32).cuda()

        # Calculate vertex position differences
        vert_diff = torch.norm(original_verts - deformed_verts, dim=1)

        # Calculate face orientation differences
        original_orien_mat, _ = compute_face_orientation(original_verts, original_faces)
        deformed_orien_mat, _ = compute_face_orientation(deformed_verts, original_faces)
        
        original_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(original_orien_mat))
        deformed_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(deformed_orien_mat))

        self.orien_diff = torch.norm(original_quat - deformed_quat, dim=1)

        
        