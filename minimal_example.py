import os

from lightning import LightningModule
from src.csdp_pipeline.factories.dataloader_factory import USleep_Dataloader_Factory
from src.csdp_training.lightning_models.factories.lightning_model_factory import (
    USleep_Factory,
)
from src.usleep.usleep import USleep


class LightningSleep(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.usleep = USleep()


class Test(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.pretrained = LightningSleep.load_from_checkpoint()


def main():
    dataloader = USleep_Dataloader_Factory(
        gradient_steps=886,
        batch_size=64,
        hdf5_base_path=os.path.join("data", "hdf5"),
        trainsets=["eesm19"],
        valsets=["eesm19"],
        testsets=["eesm19"],
        data_split_path=os.path.join("data", "splits", "test.json"),
    )

    usleep_fac = USleep_Factory(0.0001, 64, 5, 0.5, 10, 2)
    model = usleep_fac.create_pretrained_net(
        os.path.join("src", "usleep", "weights", "Depth10_CF05.ckpt")
    )

    pass


if __name__ == "__main__":
    import torch

    print(torch.__version__)  # Check PyTorch version
    print(torch.cuda.is_available())  # Should print True
    print(torch.cuda.device_count())  # Should print number of GPUs
    print(torch.cuda.get_device_name(0))  # Print GPU name
