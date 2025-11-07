# you have to run this script inside paraview.
# More docu will follow later.


import pyvista as pv
import numpy as np

from mango.util.util import to_numpy


def save_mesh(data, output_path):
    positions = data.traj_pos
    # for now only visualize ply0
    positions = to_numpy(positions[:, data.node_id == 0, :])
    faces = to_numpy(data.ply_faces)
    # save as dict
    output = {
        "positions": positions,
        "faces": faces,
    }
    # save as npz
    np.savez(output_path, **output)

    # num_faces = faces.shape[0]
    # flat_faces = np.hstack([np.full((num_faces, 1), 3), faces]).astype(int).flatten()
    #
    # # Get the number of time steps and nodes
    # num_time_steps = positions.shape[0]
    # num_nodes = positions.shape[1]
    # os.makedirs(output_path, exist_ok=True)
    #
    #
    # # Loop through each time step and create a mesh for each
    # for t in range(num_time_steps):
    #     # Get the node positions at time step t
    #     node_positions = positions[t]
    #
    #     # Create a PyVista mesh using the node positions and faces
    #     mesh = pv.PolyData(node_positions, flat_faces)
    #
    #     # Optionally, you can add additional attributes, such as node positions or other features
    #
    #     output_file = os.path.join(output_path, f"gth_{t:02}.vtp")  # Pads to 2 digits (e.g., 00, 01, 02)
    #     mesh.save(output_file)
    #
    #
    #
    # print(f"Mesh data saved to {output_path}")
