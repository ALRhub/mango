import os

import numpy as np
from torch_geometric.data import Batch

from mango.dataset.util.graph_input_output_util import get_deformable_mask, unpack_ml_batch
from mango.util.util import to_numpy

def visualize_ml_trajectories(vis_dict, epoch: int, vis_path, eval_ds):
    # create folder for this step
    step_path = os.path.join(vis_path, f"iteration_{epoch:03}")
    gth_path = os.path.join(vis_path, "ground_truth_data")
    collider_path = os.path.join(vis_path, "collider_data")
    faces_path = os.path.join(vis_path, "faces_data")
    os.makedirs(step_path, exist_ok=True)
    os.makedirs(gth_path, exist_ok=True)
    os.makedirs(collider_path, exist_ok=True)
    os.makedirs(faces_path, exist_ok=True)

    for traj_name, traj_dict in vis_dict.items():
        eval_traj = traj_dict["eval_traj"]
        predicted_traj = traj_dict["predicted_traj"]
        predicted_traj_index = traj_dict["predicted_traj_index"]
        get_gth = not os.path.exists(os.path.join(gth_path, f"{traj_name}.npz"))
        get_faces = not os.path.exists(os.path.join(faces_path, f"faces.npz"))
        # no collider for now
        get_collider = not os.path.exists(os.path.join(collider_path, f"{traj_name}.npz"))
        positions = extract_ml_positions(eval_traj, predicted_traj, predicted_traj_index, get_gth, get_collider, context_traj_index=traj_dict.get("context_traj_index", None))
        if "context_traj_index" in traj_dict and traj_dict["context_traj_index"] is not None:
            context_path = os.path.join(step_path, f"context_{traj_name}")
            os.makedirs(context_path, exist_ok=True)
            for c_idx in traj_dict["context_traj_index"]:
                c_traj = positions["context_positions"][f"deformable_{c_idx}"]
                np.savez(os.path.join(context_path, f"deformable_{c_idx}.npz"), **{"deformable": c_traj})
                if f"collider_{c_idx}" in positions["context_positions"]:
                    c_traj = positions["context_positions"][f"collider_{c_idx}"]
                    np.savez(os.path.join(context_path, f"collider_{c_idx}.npz"), **{"collider": c_traj})
        # save gth if not existent
        if get_gth:
            np.savez(os.path.join(gth_path, f"{traj_name}.npz"), **positions["gth_positions"])
        if get_collider and len(positions["collider_positions"]) > 0:
            np.savez(os.path.join(collider_path, f"{traj_name}.npz"), **positions["collider_positions"])
        if get_faces:
            faces = eval_ds.get_faces()  # faces are identical over different trajectories
            if ("cloth_identifier",) in eval_traj["h_description"]:
                # sphere cloth task, combine faces
                faces["deformable"] = np.concatenate([faces["deformable"], faces["collider"]], axis=0)
                del faces["collider"]
            np.savez(os.path.join(faces_path, f"faces.npz"), **faces)
        # save predictions in step path
        np.savez(os.path.join(step_path, f"{traj_name}.npz"), **positions["pred_positions"])

def visualize_trajectories(vis_dict, epoch: int, vis_path, eval_ds):
    # create folder for this step
    step_path = os.path.join(vis_path, f"iteration_{epoch:03}")
    gth_path = os.path.join(vis_path, "ground_truth_data")
    collider_path = os.path.join(vis_path, "collider_data")
    faces_path = os.path.join(vis_path, "faces_data")
    os.makedirs(step_path, exist_ok=True)
    os.makedirs(gth_path, exist_ok=True)
    os.makedirs(collider_path, exist_ok=True)
    os.makedirs(faces_path, exist_ok=True)

    for traj_name, traj_dict in vis_dict.items():
        eval_traj = traj_dict["eval_traj"]
        predicted_traj = traj_dict["predicted_traj"]
        get_gth = not os.path.exists(os.path.join(gth_path, f"{traj_name}.npz"))
        get_faces = not os.path.exists(os.path.join(faces_path, f"faces.npz"))
        get_collider = not os.path.exists(os.path.join(collider_path, f"{traj_name}.npz"))
        positions = extract_positions(eval_traj, predicted_traj, get_gth, get_collider)

        # save gth if not existent
        if get_gth:
            np.savez(os.path.join(gth_path, f"{traj_name}.npz"), **positions["gth_positions"])
        if get_collider:
            np.savez(os.path.join(collider_path, f"{traj_name}.npz"), **positions["collider_positions"])
        if get_faces:
            faces = eval_ds.get_faces()  # faces are identical over different trajectories
            np.savez(os.path.join(faces_path, f"faces.npz"), **faces)
        # save predictions in step pat
        np.savez(os.path.join(step_path, f"{traj_name}.npz"), **positions["pred_positions"])


def extract_ml_positions(eval_traj, predicted_traj, predicted_traj_index, get_gth=True, get_collider=True, context_traj_index=None):
    result = {}
    x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs = unpack_ml_batch(eval_traj, remove_batch_dim=True)
    if get_gth:
        # ground truth data
        gth_traj = to_numpy(x[predicted_traj_index])
        deformable_mask = h[0, :, 0] == 1
        gth_traj = gth_traj[:, deformable_mask, :]
        result["gth_positions"] = {"deformable": gth_traj}
    # predicted positions
    pred_positions = {}
    predicted_traj = to_numpy(predicted_traj)
    pred_positions["deformable"] = predicted_traj
    result["pred_positions"] = pred_positions
    # collider
    if get_collider:
        # only get a collider if it is present. this is the case if "one_hot" is 3 times in h_description
        if h_description.count("one_hot") == 3:
            gth_traj = to_numpy(x[predicted_traj_index])
            collider_mask = h[0, :, 1] == 1
            collider_traj = gth_traj[:, collider_mask, :]
            result["collider_positions"] = {"collider": collider_traj}
        else:
            result["collider_positions"] = {}
    if context_traj_index is not None:
        result["context_positions"] = {}
        for c_idx in context_traj_index:
            c_traj = to_numpy(x[c_idx])
            deformable_mask = h[0, :, 0] == 1
            c_traj_def = c_traj[:, deformable_mask, :]
            result["context_positions"][f"deformable_{c_idx}"] = c_traj_def
            if h_description.count("one_hot") == 3:
                collider_mask = h[0, :, 1] == 1
                c_traj_col = c_traj[:, collider_mask, :]
                result["context_positions"][f"collider_{c_idx}"] = c_traj_col
    return result

def extract_positions(eval_traj, predicted_traj, get_gth=True, get_collider=True):
    if isinstance(eval_traj, Batch):
        assert len(eval_traj) == 1
        eval_traj = eval_traj[0]
    result = {}
    if get_gth:
        # ground truth data
        gth_positions = {}
        for mat_idx, mat_name in eval_traj.node_id_dict.items():
            # skip collider, do it separately
            if mat_idx in eval_traj.collider_ids:
                continue
            mat_mask = eval_traj.node_id == mat_idx
            mat_positions = eval_traj.traj_pos[:, mat_mask, :]
            gth_positions[mat_name] = to_numpy(mat_positions)
        result["gth_positions"] = gth_positions
    # predicted positions
    pred_positions = {}
    deformable_mask = get_deformable_mask(eval_traj)
    deformable_node_id = eval_traj.node_id[deformable_mask]
    for mat_idx, mat_name in eval_traj.node_id_dict.items():
        # skip collider, do it separately
        if mat_idx in eval_traj.collider_ids:
            continue
        mat_mask = deformable_node_id == mat_idx
        mat_positions = predicted_traj[:, mat_mask, :]
        pred_positions[mat_name] = to_numpy(mat_positions)
    result["pred_positions"] = pred_positions
    if get_collider:
        # collider positions
        collider_positions = {}
        for mat_idx, mat_name in eval_traj.node_id_dict.items():
            if mat_idx in eval_traj.deformable_ids:
                continue
            mat_mask = eval_traj.node_id == mat_idx
            mat_positions = eval_traj.traj_pos[:, mat_mask, :]
            collider_positions[mat_name] = to_numpy(mat_positions)
        result["collider_positions"] = collider_positions
    return result
