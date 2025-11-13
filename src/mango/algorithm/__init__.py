from lightning import LightningModule

from mango.algorithm.abstract_algorithm import AbstractAlgorithm
from mango.util.own_types import ConfigDict


def get_algorithm(config: ConfigDict, train_dl, train_ds, eval_ds, loading=False, checkpoint_path=None) -> LightningModule:
    algorithm_name = config.name
    match algorithm_name:
        case "mgn":
            from mango.algorithm.mgn import MGN
            algorithm_class = MGN
        case "mango_regression":
            from mango.algorithm.mango_regression import MangoRegression
            algorithm_class = MangoRegression
        case "mango_training_mat_prop":
            from mango.algorithm.mango_training_mat_prop import MangoTrainingMatProp
            algorithm_class = MangoTrainingMatProp
        case "mango_two_stages":
            from mango.algorithm.mango_two_stages import MangoTwoStages
            algorithm_class = MangoTwoStages
        case "dummy_mango":
            from mango.algorithm.mango import Mango
            algorithm_class = Mango
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
