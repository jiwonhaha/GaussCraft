import trimesh
import numpy as np

def read_ply(file_path):
    """
    Reads a .ply file and returns a trimesh object.
    """
    mesh = trimesh.load(file_path, file_type='ply')
    return mesh

def deform_mesh(mesh, deformation_function):
    """
    Deforms the mesh according to the provided deformation function.
    """
    vertices = mesh.vertices
    deformed_vertices = deformation_function(vertices)
    mesh.vertices = deformed_vertices
    return mesh

def rotate_vertices(vertices, angle_degrees, axis):
    """
    Rotates a subset of vertices by a specified angle around a given axis.
    """
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, axis)
    
    # Example: Rotate vertices with x > 0.5
    condition = vertices[:, 0] > 0.5  # Condition to select part of the vertices
    selected_vertices = vertices[condition]
    
    # Apply rotation to selected vertices
    ones_column = np.ones((selected_vertices.shape[0], 1))
    selected_vertices_homogeneous = np.hstack([selected_vertices, ones_column])
    rotated_vertices_homogeneous = np.dot(rotation_matrix, selected_vertices_homogeneous.T).T
    rotated_vertices = rotated_vertices_homogeneous[:, :3]
    
    # Create a copy of vertices to avoid modifying the original array
    deformed_vertices = vertices.copy()
    deformed_vertices[condition] = rotated_vertices

    return deformed_vertices

def save_off(mesh, file_path):
    """
    Saves a trimesh object as a .off file.
    """
    mesh.export(file_path, file_type='off')

def main():
    input_ply = 'output/perfect/new_banana/train/ours_30000/fuse_post.ply'  # Path to your input .ply file
    output_off = 'deformed_test.off'  # Path to save the output .off file


    mesh = read_ply(input_ply)
    angle_degrees = 10  # Rotation angle in degrees
    axis = [0, 0, 1]  # Rotation axis (z-axis)
    deformed_mesh = deform_mesh(mesh, lambda vertices: rotate_vertices(vertices, angle_degrees, axis))
    save_off(deformed_mesh, output_off)

if __name__ == '__main__':
    main()
