import open3d as o3d
import numpy as np
import argparse

def perform_arap_deformation(mesh_path, handle_vertex_index, displacement_factor, num_static_vertices, reference_static_vertex_index, additional_static_ids):
    # Load the mesh from the given path
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    if not mesh.has_vertices():
        print("Failed to load the mesh. Please check the file path and format.")
        return

    # Convert mesh vertices to a NumPy array
    vertices = np.asarray(mesh.vertices)

    # Calculate the bounding box of the mesh
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_extent()

    # Calculate the displacement vector based on the bounding box size and displacement factor
    displacement_vector = displacement_factor * bbox_size

    # Define the handle vertex and its new position
    handle_position = vertices[handle_vertex_index]
    handle_ids = [handle_vertex_index]
    handle_pos = [handle_position + displacement_vector]

    # Compute distances from the reference static vertex to all other vertices
    reference_position = vertices[reference_static_vertex_index]
    distances = np.linalg.norm(vertices - reference_position, axis=1)

    # Get the indices of the closest num_static_vertices vertices
    static_ids = np.argsort(distances)[1:num_static_vertices+1].tolist()  # Exclude the reference static vertex itself
    static_pos = vertices[static_ids].tolist()

    # Add the manually specified additional static vertex group
    additional_static_pos = vertices[additional_static_ids].tolist()

    # Combine static, additional static, and handle constraints
    constraint_ids = o3d.utility.IntVector(static_ids + additional_static_ids + handle_ids)
    constraint_pos = o3d.utility.Vector3dVector(static_pos + additional_static_pos + handle_pos)

    # Perform ARAP deformation with verbosity level set to Debug
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_prime = mesh.deform_as_rigid_as_possible(
            constraint_ids, constraint_pos, max_iter=50
        )

    # Save the deformed mesh to a file
    output_path = "deformed_mesh.ply"
    o3d.io.write_triangle_mesh(output_path, mesh_prime)
    print(f"Deformed mesh saved to {output_path}")

    # Optionally, visualize the deformed mesh
    o3d.visualization.draw_geometries([mesh_prime])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform ARAP deformation on a mesh.")
    parser.add_argument("mesh_path", type=str, help="Path to the input mesh file.")
    parser.add_argument("handle_vertex_index", type=int, nargs='?', default=2490, help="Index of the handle vertex to be moved (default: 2490).")
    parser.add_argument("displacement_factor", type=float, nargs='?', default=0.1, help="Factor to scale the displacement based on the mesh size (default: 0.1).")
    parser.add_argument("num_static_vertices", type=int, nargs='?', default=10, help="Number of static vertices (default: 10).")
    parser.add_argument("reference_static_vertex_index", type=int, nargs='?', default=0, help="Index of the reference static vertex (default: 0).")
    parser.add_argument("additional_static_ids", type=int, nargs='+', help="List of additional static vertex indices.")
    args = parser.parse_args()

    perform_arap_deformation(args.mesh_path, args.handle_vertex_index, args.displacement_factor, args.num_static_vertices, args.reference_static_vertex_index, args.additional_static_ids)
