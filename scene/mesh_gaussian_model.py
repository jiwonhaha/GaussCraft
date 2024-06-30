import torch
from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation

class mesh_gaussian_model(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self.verts = None
        self.faces = None

    def load_mesh(self, vertices, faces):
        """Load a single general triangular mesh."""
        self.verts = torch.tensor(vertices, dtype=torch.float32).cuda()
        self.faces = torch.tensor(faces, dtype=torch.int32).cuda()
        
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
        # self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))


    def training_setup(self, training_args):
        """Setup the optimizer for the general mesh."""
        super().training_setup(training_args)
        self.verts.requires_grad = True
        params = [{'params': self.verts, 'lr': training_args.mesh_lr, "name": "mesh_verts"}]
        self.optimizer.add_param_group(params)

    def save_ply(self, path):
        """Save the mesh and Gaussian parameters to a file."""
        super().save_ply(path)
        # Additional saving logic if necessary

    def load_ply(self, path, **kwargs):
        """Load the mesh and Gaussian parameters from a file."""
        super().load_ply(path)
        # Additional loading logic if necessary
