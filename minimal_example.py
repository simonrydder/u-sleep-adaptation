import os

from src.csdp_pipeline.factories.dataloader_factory import USleep_Dataloader_Factory
from src.csdp_training.lightning_models.factories.lightning_model_factory import (
    USleep_Factory,
)


def main():
    dataloader = USleep_Dataloader_Factory(
        gradient_steps=100,
        batch_size=64,
        hdf5_base_path=os.path.join("data", "hdf5"),
        trainsets=["eesm19"],
        valsets=["eesm19"],
        testsets=["eesm19"],
        data_split_path=os.path.join("data", "splits", "test.json"),
    )

    pass
    test_loader = dataloader.create_testing_loader(1)

    model = USleep_Factory(0.0001, 64, 10, 0.5, 2)


if __name__ == "__main__":
    main()
