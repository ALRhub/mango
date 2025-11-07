# start paraview with venv support using the command:
# paraview --venv /home/philipp/projects/stamp_forming_sim/paraview_venv_2

# import matplotlib
import glob
import shutil
import trimesh
import numpy as np
import os
import colorsys
import hashlib

TEMP_DIR = "/home/philipp/projects/stamp_forming_sim/output/tmp"


# delete folder and create new one
# if os.path.exists(TEMP_DIR):
#     shutil.rmtree(TEMP_DIR)
# os.makedirs(TEMP_DIR, exist_ok=True)


def load_mesh(all_def_positions, all_faces, gth=False, base_folder="example_traj"):
    face_dict = {
        "ply0": "ply_tri",
        "ply45": "ply_tri",
        "ply90": "ply_tri",
        "plym45": "ply_tri",
    }
    for key, def_pos in all_def_positions.items():
        if def_pos.shape[-1] == 2:
            # add z dimension
            def_pos = np.concatenate([def_pos, np.zeros((def_pos.shape[0], def_pos.shape[1], 1))], axis=-1)
        if key in face_dict:
            face_key = face_dict[key]
        else:
            face_key = key
        face = all_faces[face_key]
        if face.shape[1] == 4:
            # tetrahedral faces, draw floor since it is the torus task
            draw_floor = True
        else:
            draw_floor = False
        # Get the number of time steps and nodes
        num_time_steps = def_pos.shape[0]

        # Export to a .ply file
        if gth:
            output_folder = os.path.join(TEMP_DIR, base_folder, "gth", key)
        else:
            output_folder = os.path.join(TEMP_DIR, base_folder, "pred", key)
        # if exists, delete and create new folder
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        # Loop through each time step and create a mesh for each
        for t in range(num_time_steps):
            # Get the node positions at time step t
            node_positions = def_pos[t]
            # Create a Trimesh object with the node positions and faces
            mesh = trimesh.Trimesh(vertices=node_positions, faces=face)
            out_file = f"{t:03}.ply"
            output_file = os.path.join(output_folder, out_file)  # Pads to 2 digits (e.g., 00, 01, 02)
            mesh.export(output_file)
    if gth:
        print("Done creating gth files")
    else:
        print("Done creating pred files")
    return draw_floor


def load_collider(all_collider_pos, all_faces, base_folder="example_traj"):
    face_dict = {
        "bottom_tool": ["bottom_tool_quad", "bottom_tool_tri"],
        "top_tool": ["top_tool_quad", "top_tool_tri"],
    }
    for key, collider_pos in all_collider_pos.items():
        if collider_pos.shape[-1] == 2:
            # add z dimension
            collider_pos = np.concatenate([collider_pos, np.zeros((collider_pos.shape[0], collider_pos.shape[1], 1))], axis=-1)
        # Get the number of time steps and nodes
        num_time_steps = collider_pos.shape[0]
        if key in face_dict:
            face_key = face_dict[key]
        else:
            face_key = key
        if not isinstance(face_key, list):
            face_key = [face_key]
        combined_faces = []
        for face_type in face_key:
            face = all_faces[face_type]
            if face.shape[-1] == 4:
                # Convert quad faces to triangle faces
                quad_as_triangles = []
                for quad in face:
                    quad_as_triangles.append([quad[0], quad[1], quad[2]])  # First triangle
                    quad_as_triangles.append([quad[0], quad[2], quad[3]])  # Second triangle
                quad_as_triangles = np.array(quad_as_triangles)
                combined_faces.append(quad_as_triangles)
            else:
                combined_faces.append(face)
        face = np.concatenate(combined_faces, axis=0)

        # Export to a .ply file
        output_folder = os.path.join(TEMP_DIR, base_folder, "collider", key)
        # if exists, delete and create new folder
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        # Loop through each time step and create a mesh for each
        for t in range(num_time_steps):
            # Get the node positions at time step t
            node_positions = collider_pos[t]

            # Create a Trimesh object with the node positions and faces
            mesh = trimesh.Trimesh(vertices=node_positions, faces=face)

            out_file = f"{t:03}.ply"
            output_file = os.path.join(output_folder, out_file)  # Pads to 2 digits (e.g., 00, 01, 02)
            mesh.export(output_file)
    print("Done creating collider files")


def load_floor():
    from paraview.simple import Plane, GetActiveViewOrCreate, Show, Render, ColorBy, GetColorTransferFunction, Calculator

    # Create a new plane source
    plane = Plane()

    # Set the properties of the plane
    plane.Origin = [-1, -1, 0]
    plane.Point1 = [1, -1, 0]
    plane.Point2 = [-1, 1, 0]

    # Show the plane in the render view
    renderView = GetActiveViewOrCreate('RenderView')
    planeDisplay = Show(plane, renderView)

    # Set representation to Surface (optional)
    planeDisplay.Representation = 'Surface'
    # Add a gradient color using one of the axes (e.g., Y-coordinate)
    ColorBy(planeDisplay, ('POINTS', 'Coords', 'Y'))

    # Rescale color to the data range
    planeDisplay.RescaleTransferFunctionToDataRange(True)

    calculator = Calculator(Input=plane)
    calculator.Function = "coordsY + coordsX"

    # Show the calculated gradient data
    gradientDisplay = Show(calculator, renderView)
    gradientDisplay.Representation = 'Surface'

    # Color by the calculated variable (e.g., Result from the calculator)
    ColorBy(gradientDisplay, ('POINTS', 'Result'))

    # Rescale color map to fit the data range
    gradientDisplay.RescaleTransferFunctionToDataRange(True, False)

    # Use a preset for better visualization (e.g., "Cool to Warm")
    lut = GetColorTransferFunction('Result')
    lut.ApplyPreset('Cool to Warm', True)


def show_in_paraview(base_folder="example_traj", color=None, hide=False):
    from paraview.simple import PLYReader, GetAnimationScene, GetActiveViewOrCreate, Show, Hide
    base_path = os.path.join(TEMP_DIR, base_folder)
    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    renderView1.ResetCamera(False)
    renderView1.InteractionMode = '3D'
    # camera pos for planar bending. have to make more general with different datasets
    renderView1.CameraPosition = [0.32063086865095286, -0.9122212806202091,
                                  0.5937422430893627]  # Replace with extracted CameraPosition
    renderView1.CameraFocalPoint = [0.4866583183660358, 0.4914762389911497,
                                    -0.02299186532575726]  # Replace with extracted CameraFocalPoint
    renderView1.CameraViewUp = [-0.010846570714784743, 0.4033020923245255,
                                0.9150026088653458]  # Replace with extracted CameraViewUp

    for dirpath, dirnames, filenames in os.walk(base_path):
        print(f"Directory: {dirpath}")
        obj_type = dirpath.split("/")[-2]
        obj_name = dirpath.split("/")[-1]
        if obj_type in ["gth", "pred", "collider"]:
            # Get sorted lists of .vtp files
            mesh_files = sorted(filenames)
            full_path_mesh_files = [os.path.join(dirpath, file) for file in mesh_files]
            # load if path exists
            if not full_path_mesh_files:
                print(f"No files found in {dirpath}")
                continue
            full_object_name = obj_type + "_" + obj_name + "_" + base_folder
            # pred_files = sorted(glob.glob(os.path.join(pred_path, "*.vtp")))
            # Load gth (ground truth)
            mesh = PLYReader(registrationName=f'{full_object_name}', FileNames=full_path_mesh_files)
            # get animation scene
            animationScene1 = GetAnimationScene()

            # update animation scene based on data timesteps
            animationScene1.UpdateAnimationUsingDataTimeSteps()

            # show data in view
            mesh_display = Show(mesh, renderView1, 'GeometryRepresentation')
            if obj_type == "gth":
                mesh_display.SetRepresentationType('Wireframe')
            elif obj_type == "pred":
                # Apply color to the prediction mesh
                if color is not None:
                    mesh_display.DiffuseColor = color  # Example: green color
            if hide:
                Hide(mesh, renderView1)


def string_to_color(string):
    """
    Generate a consistent color for a string using HSL.
    """
    # Hash the string to a consistent value
    hash_value = int(hashlib.md5(string.encode()).hexdigest(), 16)

    # Map the hash to a hue value (0-360 degrees)
    hue = (hash_value % 360) / 360.0  # Normalize to 0–1 for colorsys
    saturation = 1.0  # Full saturation
    lightness = 0.5  # Medium lightness

    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (r, g, b)


def load_visualization(iter_path, traj_idx, hide=False, process_only=False):
    vis_path = os.path.dirname(iter_path)
    assert vis_path.endswith(
        "visualizations"), "The path to the iteration folder must be inside the visualizations folder"
    # Extract the directory name and iteration
    dir_name = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(iter_path)))))
    color = string_to_color(dir_name)
    iteration = os.path.basename(os.path.normpath(iter_path))
    # Combine to create the desired name
    base_folder = f"{dir_name}__{iteration}"
    base_folder = base_folder + f"__Traj_{traj_idx}"
    print(base_folder)

    # load context
    context_folder = os.path.join(iter_path, f"context_Traj_{traj_idx}")
    to_show = True
    for file_name in sorted(os.listdir(context_folder)):
        if "collider" in file_name:
            continue
        context_idx = file_name.split("_")[-1].split(".")[0]
        context_base_folder = base_folder + f"__Context_{context_idx}"
        all_def_positions = np.load(os.path.join(context_folder, file_name))
        all_faces = np.load(os.path.join(vis_path, "faces_data", "faces.npz"), allow_pickle=True)
        load_mesh(all_def_positions, all_faces, gth=False, base_folder=context_base_folder)
        # load collider
        # only load collider if file exists
        collider_file_name = os.path.join(context_folder, f"collider_{context_idx}.npz")
        if os.path.exists(collider_file_name):
            all_collider_positions = np.load(collider_file_name)
            load_collider(all_collider_positions, all_faces, base_folder=context_base_folder)

        if not process_only:
            # now all mesh files are created, load them in paraview
            show_in_paraview(base_folder=context_base_folder, color=color, hide=not to_show)
            to_show = False
    return draw_floor


# Example usage
process_only = False  # make it true if only conversion for blender needs to be done
iter_path = input("Enter the path to the iteration folder inside the visualization folder: ")
# iter_path = "/home/philipp/projects/stamp_forming_sim/output/hydra/training/2024-12-19/p9999_double_dome_playground/visualizations/iteration_001/"
if iter_path.endswith("/"):
    iter_path = iter_path[:-1]
traj_idx = input(
    "Enter the index of the trajectory to visualize. Enter a single blank space or 'all' to plot all trajs: ")
print("Traj_idx: ", traj_idx)
# traj_idx = "0"
draw_floor = load_visualization(iter_path, traj_idx, process_only=process_only)

if draw_floor and not process_only:
    # load a floor
    load_floor()
