import open3d as o3d
import numpy as np
import argparse

def perform_arap_deformation(mesh, handle_vertex_indices, handle_displacement, static_vertex_indices):
    # Convert mesh vertices to a NumPy array
    vertices = np.asarray(mesh.vertices)

    # Apply the displacement to all handle vertices
    displacement_vector = np.array(handle_displacement)
    handle_pos = [vertices[i] + displacement_vector for i in handle_vertex_indices]

    # Retrieve positions of the static vertices
    static_pos = vertices[static_vertex_indices].tolist()

    # Combine static and handle constraints
    constraint_ids = o3d.utility.IntVector(static_vertex_indices + handle_vertex_indices)
    constraint_pos = o3d.utility.Vector3dVector(static_pos + handle_pos)

    # Perform ARAP deformation with verbosity level set to Debug
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_prime = mesh.deform_as_rigid_as_possible(
            constraint_ids, constraint_pos, max_iter=1000
        )

    return mesh_prime

def select_vertices(vis, mesh, message):
    print(message)
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()  # This will allow user to select vertices
    vis.destroy_window()

    # Get the selected vertices
    selected = vis.get_picked_points()
    vertex_indices = [p.index for p in selected]

    print(f"Selected vertices: {vertex_indices}")
    return vertex_indices

def select_and_deform(mesh):
    # Select handle vertices
    handle_vertex_indices = select_vertices(None, mesh, "Select handle vertices.")

    if not handle_vertex_indices:
        print("No handle vertices selected.")
        return

    # Select static vertices
    static_vertex_indices = select_vertices(None, mesh, "Select static vertices.")

    if not static_vertex_indices:
        print("No static vertices selected.")
        return

    # Ask the user for displacement input for handle vertices
    displacement = input("Enter displacement vector as x,y,z (e.g., 1,0,0): ")
    displacement = [float(x) for x in displacement.split(',')]

    # Perform ARAP deformation
    deformed_mesh = perform_arap_deformation(
        mesh,
        handle_vertex_indices,
        displacement,
        static_vertex_indices
    )

    # Save the deformed mesh to a file
    output_path = "deformed_mesh.ply"
    o3d.io.write_triangle_mesh(output_path, deformed_mesh)
    print(f"Deformed mesh saved to {output_path}")

    # Visualize the deformed mesh
    o3d.visualization.draw_geometries([deformed_mesh])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform ARAP deformation on a mesh with interactive vertex selection.")
    parser.add_argument("mesh_path", type=str, help="Path to the input mesh file.")
    args = parser.parse_args()

    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(args.mesh_path)
    
    if not mesh.has_vertices():
        print("Failed to load the mesh. Please check the file path and format.")
    else:
        select_and_deform(mesh)
