import os
import pickle

import h5py
import numpy as np
from tqdm import tqdm


def create_quad_mesh():
    # Define grid size
    grid_size = 20

    # Generate quad faces
    quad_faces = []
    for y in range(grid_size - 1):  # Iterate over rows
        for x in range(grid_size - 1):  # Iterate over columns
            # Compute indices of the four corners of the quad
            top_left = y * grid_size + x
            top_right = top_left + 1
            bottom_left = top_left + grid_size
            bottom_right = bottom_left + 1

            # Append the face as a 4-tuple
            quad_faces.append((top_left, top_right, bottom_right, bottom_left))

    # Convert the list of 4-tuples to a numpy array
    quad_faces = np.array(quad_faces, dtype=np.int32)
    return quad_faces


def create_triangle_mesh():
    grid_size = 20

    # Generate triangles
    triangles = []
    for y in range(grid_size - 1):  # Iterate over rows
        for x in range(grid_size - 1):  # Iterate over columns
            # Compute indices of the four corners of the quad
            top_left = y * grid_size + x
            top_right = top_left + 1
            bottom_left = top_left + grid_size
            bottom_right = bottom_left + 1

            # Split quad into two triangles
            triangles.append((top_left, top_right, bottom_right))  # First triangle
            triangles.append((top_left, bottom_right, bottom_left))
    triangles = np.array(triangles, dtype=np.int32)
    return triangles


def create_hd5f_deformable_plate(input_path: str, output_path: str, ):
    # load pickle
    data = {
        "sphere_positions": [],
        "sphere_velocities": [],
        "sphere_density": [],
        "cloth_positions": [],
        "cloth_velocities": [],
    }
    # for file2 in sorted(os.listdir(input_path)):
    #     print(f"Loading file {file2}...")
    #     with open(os.path.join(input_path, file2), "rb") as file:
    #         data = pickle.load(file)
    #         data['sphere_density'] = np.array([data['sphere_density']])
    #     break

    # generate mesh faces
    faces = create_triangle_mesh()

    # # Plot 3d points
    # mesh = data["cloth_positions"][0, 0]
    # plt.figure(figsize=(10, 7))
    # ax = plt.axes(projection='3d')
    #
    # ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], label="Mesh Points", color="blue")
    # # # Label points
    # # for idx, (x, y, z) in enumerate(mesh):
    # #     ax.text(x, y, z, str(idx), fontsize=8, ha='right', va='bottom', color='red')
    # # plot faces
    # for face in faces:
    #     face = np.append(face, face[0])
    #     ax.plot(mesh[face, 0], mesh[face, 1], mesh[face, 2], color="black")
    #
    # # Add legend and display
    # plt.legend()
    # # plt.title("Mesh and Collider Points with Connection Edges")
    # # plt.xlabel("X")
    # # plt.ylabel("Y")
    # plt.grid(True)
    # plt.show()

    for file in sorted(os.listdir(input_path)):
        print(f"Loading file {file}...")
        with open(os.path.join(input_path, file), "rb") as file:
            file_data = pickle.load(file)
            data["sphere_positions"].append(file_data["sphere_positions"])
            data["sphere_velocities"].append(file_data["sphere_velocities"])
            data["sphere_density"].append(np.array([file_data["sphere_density"]]))
            data["cloth_positions"].append(file_data["cloth_positions"])
            data["cloth_velocities"].append(file_data["cloth_velocities"])
            data["sphere_indices"] = file_data["sphere_indices"]
    # concatenate
    # data["sphere_positions"] = np.concatenate(data["sphere_positions"], axis=1)
    # data["sphere_velocities"] = np.concatenate(data["sphere_velocities"], axis=1)
    # data["sphere_density"] = np.concatenate(data["sphere_density"], axis=0)
    # data["cloth_positions"] = np.concatenate(data["cloth_positions"], axis=1)
    # data["cloth_velocities"] = np.concatenate(data["cloth_velocities"], axis=1)
    #
    #
    # data["sphere_positions"] = data["sphere_positions"].transpose(1, 0, 2, 3)
    # data["sphere_velocities"] = data["sphere_velocities"].transpose(1, 0, 2, 3)
    # data["cloth_positions"] = data["cloth_positions"].transpose(1, 0, 2, 3)
    # data["cloth_velocities"] = data["cloth_velocities"].transpose(1, 0, 2, 3)
    print("Data loaded from files.")
    np.random.seed(42)
    total_samples = 800
    n_train = 600
    n_test = 100
    n_val = 100
    task_size = 16
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
        for idx in tqdm(range(total_samples), "Adding Tasks"):
            traj_start_index = idx * task_size
            traj_end_index = (idx + 1) * task_size
            # task_data = {
            #     "vertices_trajectory": data["vertices_trajectory"][traj_start_index:traj_end_index],
            #     "young_modulus": data["youngsModulus"][traj_start_index:traj_end_index],
            #     "poisson_ratio": data["poissonsRatio"][traj_start_index:traj_end_index],
            #     "friction": data["frictions"][traj_start_index:traj_end_index],
            # }
            task_data = {
                "sphere_positions": data["sphere_positions"][idx],
                "sphere_velocities": data["sphere_velocities"][idx],
                "sphere_density": data["sphere_density"][idx],
                "cloth_positions": data["cloth_positions"][idx],
                "cloth_velocities": data["cloth_velocities"][idx],
            }
            task_data["sphere_positions"] = task_data["sphere_positions"].transpose(1, 0, 2, 3)
            task_data["sphere_velocities"] = task_data["sphere_velocities"].transpose(1, 0, 2, 3)
            task_data["cloth_positions"] = task_data["cloth_positions"].transpose(1, 0, 2, 3)
            task_data["cloth_velocities"] = task_data["cloth_velocities"].transpose(1, 0, 2, 3)

            per_task = f.create_group(f"task_{idx:03}")
            params = per_task.create_group("params")
            trajs = per_task.create_group("trajs")
            # check that all yms are the same
            # assert len(np.unique(task_data["sphere_density"])) == 1
            params.create_dataset("sphere_density", data=task_data["sphere_density"])
            # create trajs data
            for traj_idx in range(task_size):
                traj_group = trajs.create_group(f"traj_{traj_idx:03}")
                # position data
                traj_group.create_dataset("sphere_positions", data=task_data["sphere_positions"][traj_idx])
                traj_group.create_dataset("cloth_positions", data=task_data["cloth_positions"][traj_idx])
                # velocity data, not needed, so don't save it to reduce space
                # traj_group.create_dataset("sphere_velocities", data=task_data["sphere_velocities"][traj_idx])
                # traj_group.create_dataset("cloth_velocities", data=task_data["cloth_velocities"][traj_idx])

        # add global data
        global_data = f.create_group("global_data")
        cell_indices = global_data.create_group("cell_indices")
        cell_indices.create_dataset("sphere_indices", data=data["sphere_indices"].reshape(-1, 3))
        cell_indices.create_dataset("cloth_indices", data=faces)

    print("HDF5 file created with the required structure.")


if __name__ == "__main__":
    output_path = f"/home/philipp/lsdf/for5339/datasets/ltsgnsv2/sphere_cloth_v3.hdf5"
    create_hd5f_deformable_plate(f"/home/philipp/lsdf/for5339/datasets/ltsgnsv2/sphere_cloth_coupling_v3",
                                 output_path)
