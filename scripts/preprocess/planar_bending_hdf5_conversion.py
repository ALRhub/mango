import os

import h5py
import numpy as np
import yaml
import meshio
from tqdm import tqdm



def check_data(input_path):
    for file in os.listdir(input_path):
        for sub_file in os.listdir(os.path.join(input_path, file)):
            assert "PLY-1" in os.listdir(os.path.join(input_path, file,
                                                      sub_file)), f"PLY-1 not found in {os.path.join(input_path, file, sub_file)}"
            # check number of files in PLY-1
            assert len(os.listdir(os.path.join(input_path, file, sub_file, "PLY-1"))) == 102, f"Not correct files in {os.path.join(input_path, file, sub_file, 'PLY-1')}"
    print("Data is consistent.")


def create_hd5f_planar_bending(input_path: str, output_path: str, ):
    np.random.seed(42)
    total_samples = 560
    n_train = 460
    n_test = 50
    n_val = 50
    indices = np.arange(total_samples)
    # shuffle
    np.random.shuffle(indices)
    # Create an HDF5 file
    with h5py.File(output_path, "w") as f:
        # Splits Group
        splits = f.create_group("splits")
        splits.create_dataset("train_indices", data=indices[:n_train])
        splits.create_dataset("test_indices", data=indices[n_train:n_train + n_test])
        splits.create_dataset("val_indices", data=indices[n_train + n_test:])
        for idx, folder in tqdm(enumerate(sorted(os.listdir(input_path))), "Adding Tasks", disable=True):
            print("Processing task", idx, " ,Folder", folder)
            per_task = f.create_group(f"task_{idx:03}")
            params = per_task.create_group("params")
            trajs = per_task.create_group("trajs")
            with open(os.path.join(input_path, folder, "Sim000", "Parameter.yaml"), "r") as file:
                parameters = yaml.safe_load(file)
                params.create_dataset("youngs_modulus",
                                      data=np.array(parameters["params"]["material"]["YoungsModulus"]))
            # create trajs data
            for traj_idx, traj_folder in enumerate(sorted(os.listdir(os.path.join(input_path, folder)))):
                print("    processing traj", traj_idx, " ,Folder", traj_folder)
                traj_group = trajs.create_group(f"traj_{traj_idx:03}")
                # force params
                traj_params = traj_group.create_group("params")
                with open(os.path.join(input_path, folder, traj_folder, "Parameter.yaml"), "r") as file:
                    parameters = yaml.safe_load(file)
                    for force in parameters["params"]["forces"].keys():
                        force_group = traj_params.create_group(force)
                        force_group.create_dataset("direction", data=parameters["params"]["forces"][force]["direction"])
                        force_group.create_dataset("position", data=parameters["params"]["forces"][force]["position"])
                # position data

                # process ply folder
                ply_folder = os.path.join(input_path, folder, traj_folder, "PLY-1")
                combined_pos = []
                for file in sorted(os.listdir(ply_folder), key=lambda x: int(x[x.rfind("-") + 1:x.rfind(".")])):
                    # process step file
                    step = file[file.rfind("-") + 1:file.rfind(".")]
                    if file.endswith(".vtk"):
                        # load with meshio
                        mesh = meshio.read(os.path.join(ply_folder, file))
                        mesh_pos = mesh.points
                        combined_pos.append(mesh_pos)
                combined_pos = np.stack(combined_pos)
                traj_group.create_dataset("mesh_pos", data=combined_pos)
        # add global data
        global_data = f.create_group("global_data")
        for idx, folder in enumerate(sorted(os.listdir(input_path))):
            ply_file = os.path.join(input_path, folder, "Sim000", "PLY-1", "PLY-1-0.vtk")
            mesh = meshio.read(ply_file)
            global_data.create_dataset("cells", data=mesh.cells_dict["triangle"])
            break

    print("HDF5 file created with the required structure.")


def convert_to_node_feature(mesh, feature_name):
    cell_data = mesh.cell_data[feature_name][0]
    cell_types = list(mesh.cells_dict.keys())
    cells = mesh.cells_dict[cell_types[0]]
    node_data = np.zeros((mesh.points.shape[0], 1))
    node_occurrences = np.zeros((mesh.points.shape[0], 1))

    # Flatten the cells and repeat the corresponding cell data
    nodes = cells.flatten()  # Flatten all nodes from the cells
    cell_indices = np.repeat(np.arange(len(cells)), cells.shape[1])  # Repeat the cell indices for each node

    # Update node_data and node_occurrences
    node_data[nodes] += cell_data[cell_indices].reshape(-1, 1)  # Add cell data to corresponding nodes
    node_occurrences[nodes] += 1  # Count occurrences of each node

    # Average the node data
    assert np.all(node_occurrences > 0)
    node_data = node_data / node_occurrences
    return node_data


if __name__ == "__main__":
    # create_hd5f_planar_bending("/home/philipp/projects/gymenvironment/example/02_PlateDefo_BI/results",
    #                         "/home/philipp/lsdf/for5339/datasets/ltsgnsv2/ml_pb_v1.hdf5")
    # load_hdf5("/home/philipp/lsdf/for5339/datasets/ltsgnsv2/ml_pb_v1.hdf5")
    input_path = "/home/philipp/projects/gymenvironment/example/02_PlateDefo_BI/results/ml_pb_hard_v1"
    print("Checking data")
    check_data(input_path)
    create_hd5f_planar_bending(input_path,
                               "/home/philipp/lsdf/for5339/datasets/ltsgnsv2/ml_pb_hard_v1.hdf5")
    # load_hdf5("/home/philipp/lsdf/for5339/datasets/ltsgnsv2/ml_pb_v2.hdf5")
