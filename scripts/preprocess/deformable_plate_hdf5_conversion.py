import pickle

import h5py
import numpy as np
from tqdm import tqdm



def create_hd5f_deformable_plate(input_path: str, output_path: str, ):
    # load pickle
    with open(input_path, "rb") as file:
        data = pickle.load(file)
    np.random.seed(42)
    total_samples = 800
    n_train = 600
    n_test = 100
    n_val = 100
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
        for idx, task in tqdm(enumerate(data), "Adding Tasks"):
            per_task = f.create_group(f"task_{idx:03}")
            params = per_task.create_group("params")
            trajs = per_task.create_group("trajs")
            # check that all yms are the same
            ym = task[0]["young_modulus"]
            assert all([t["young_modulus"] == ym for t in task])
            params.create_dataset("youngs_modulus", data=task[0]["young_modulus"])
            # poisson ratio
            poisson_ratio = task[0]["poisson_ratio"]
            assert all([t["poisson_ratio"] == poisson_ratio for t in task])
            params.create_dataset("poisson_ratio", data=task[0]["poisson_ratio"])
            # create trajs data
            for traj_idx, traj in enumerate(task):
                traj_group = trajs.create_group(f"traj_{traj_idx:03}")
                # position data
                traj_group.create_dataset("mesh_pos", data=np.stack(traj["nodes_grid"]))
                traj_group.create_dataset("collider_pos", data=np.stack(traj["nodes_collider"]))

        # add global data
        global_data = f.create_group("global_data")
        cell_indices = global_data.create_group("cell_indices")
        for idx, task in enumerate(data):
            traj = task[0]
            cell_indices.create_dataset("mesh_faces", data=traj["triangles_grid"])
            cell_indices.create_dataset("collider_faces", data=traj["triangles_collider"])
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
    for input_file, output_file in zip(["dp_ml_easy_5", "dp_ml_hard_5"], ["ml_dp_easy_v5.hdf5", "ml_dp_hard_v5.hdf5"]):
        output_path = f"/home/philipp/lsdf/for5339/datasets/ltsgnsv2/{output_file}"
        create_hd5f_deformable_plate(f"/home/philipp/projects/CloudDeform/clouddeform/sofa/data/{input_file}/{input_file}_train.pkl",
                                   output_path)
    # load_hdf5(output_path)
