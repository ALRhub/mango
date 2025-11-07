from lightning import LightningModule

from mango.algorithm.abstract_algorithm import AbstractAlgorithm
from mango.util.own_types import ConfigDict


def get_algorithm(config: ConfigDict, train_dl, train_ds, eval_ds, loading=False, checkpoint_path=None) -> LightningModule:
    algorithm_name = config.name
    match algorithm_name:
        case "mgn":
            from mango.algorithm.mgn import MGN
            algorithm_class = MGN
        case "ltsgns_v2":
            from mango.algorithm.ltsgns_v2 import LTSGNSV2
            algorithm_class = LTSGNSV2
        case "ltsgns_v2_regression":
            from mango.algorithm.ltsgns_v2_regression import LTSGNSV2Regression
            algorithm_class = LTSGNSV2Regression
        case "ltsgns_v2_training_mat_prop":
            from mango.algorithm.ltsgns_v2_training_mat_prop import LTSGNSV2TrainingMatProp
            algorithm_class = LTSGNSV2TrainingMatProp
        case "ltsgns_v2_two_stages":
            from mango.algorithm.ltsgns_v2_two_stages import LTSGNSV2TwoStages
            algorithm_class = LTSGNSV2TwoStages
        case "no_ml_mgn_torch_geometric":
            from mango.algorithm.no_ml_mgn_torch_geometric import NoMLMGNTorchGeometric
            algorithm_class = NoMLMGNTorchGeometric
        case "no_ml_mgn":
            from mango.algorithm.no_ml_mgn import NoMLMGN
            algorithm_class = NoMLMGN
        case _:
            raise ValueError(f"Unknown algorithm {algorithm_name}")

    if loading:
        algorithm = algorithm_class.load_from_checkpoint(checkpoint_path, train_dl=train_dl, train_ds=train_ds, eval_ds=eval_ds)
    else:
        algorithm = algorithm_class(config, train_dl, train_ds, eval_ds)

    return algorithm
