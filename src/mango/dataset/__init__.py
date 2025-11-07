def get_dataset(config, loading=False):
    eval_split = "val" if not loading else "test"
    match config.name:
        case "double_dome":
            from mango.dataset.step_datasets.double_dome_dataset import DoubleDomeDataset
            from mango.dataset.traj_datasets.traj_double_dome_dataset import TrajDoubleDomeDataset
            train_ds = DoubleDomeDataset(config.train_dataset, split="train")
            eval_ds = TrajDoubleDomeDataset(config.eval_dataset, split=eval_split)
        case "pb_ml":
            from mango.dataset.ml_datasets.planar_bending import PlanarBendingDataset
            train_ds = PlanarBendingDataset(config.train_dataset, split="train")
            eval_ds = PlanarBendingDataset(config.eval_dataset, split=eval_split)
        case "pb_step_ml":
            from mango.dataset.step_datasets.step_planar_bending import StepPlanarBendingDataset
            from mango.dataset.ml_datasets.planar_bending import PlanarBendingDataset
            train_ds = StepPlanarBendingDataset(config.train_dataset, split="train")
            eval_ds = PlanarBendingDataset(config.eval_dataset, split=eval_split)
        case "dp_ml":
            from mango.dataset.ml_datasets.deformable_plate import DeformablePlateDataset
            train_ds = DeformablePlateDataset(config.train_dataset, split="train")
            eval_ds = DeformablePlateDataset(config.eval_dataset, split=eval_split)
        case "dp_step_ml":
            from mango.dataset.step_datasets.step_deformable_plate import StepDeformablePlateDataset
            from mango.dataset.ml_datasets.deformable_plate import DeformablePlateDataset
            train_ds = StepDeformablePlateDataset(config.train_dataset, split="train")
            eval_ds = DeformablePlateDataset(config.eval_dataset, split=eval_split)
        case "torus_ml":
            from mango.dataset.ml_datasets.torus import TorusDataset
            train_ds = TorusDataset(config.train_dataset, split="train")
            eval_ds = TorusDataset(config.eval_dataset, split=eval_split)
        case "torus_step_ml":
            from mango.dataset.step_datasets.step_torus import StepTorusDataset
            from mango.dataset.ml_datasets.torus import TorusDataset
            train_ds = StepTorusDataset(config.train_dataset, split="train")
            eval_ds = TorusDataset(config.eval_dataset, split=eval_split)
        case "sphere_cloth_ml":
            from mango.dataset.ml_datasets.sphere_cloth import SphereClothDataset
            train_ds = SphereClothDataset(config.train_dataset, split="train")
            eval_ds = SphereClothDataset(config.eval_dataset, split=eval_split)
        case "sphere_cloth_step_ml":
            from mango.dataset.step_datasets.step_sphere_cloth import StepSphereClothDataset
            from mango.dataset.ml_datasets.sphere_cloth import SphereClothDataset
            train_ds = StepSphereClothDataset(config.train_dataset, split="train")
            eval_ds = SphereClothDataset(config.eval_dataset, split=eval_split)
        case _:
            raise ValueError(f"Dataset {config.name} unknown.")
    return train_ds, eval_ds
