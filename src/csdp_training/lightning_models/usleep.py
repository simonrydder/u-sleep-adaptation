"""Usleep wrapped in lightning module, based on a base class"""


# Code inspired by U-Sleep article
# and https://github.com/neergaard/utime-pytorch

# pylint: disable=missing-function-docstring,invalid-name

import torch

from src.csdp_training.lightning_models.base import Base_Lightning
from src.csdp_training.utility import log_test_step
from src.usleep import USleep


class USleep_Lightning(Base_Lightning):
    """lightning wrapper for the usleep network class

    Forward pass expects the following output:
    x_eeg: torch.Tensor
        EEG signal
    x_eog: torch.Tensor
        EOG signal
    ybatch: torch.Tensor
        target labels
    tags: list
        ID tags for a given set of epochs, to make it easier to identify the results (mostly for debugging purposes)
    """

    def __init__(
        self,
        lr,
        batch_size,
        initial_filters,
        complexity_factor,
        progression_factor,
        lr_patience,
        lr_factor,
        lr_minimum,
        loss_weights,
        include_eog=True,
    ):
        num_channels = 2 if include_eog is True else 1

        inner = USleep(
            num_channels=num_channels,
            initial_filters=initial_filters,
            complexity_factor=complexity_factor,
            progression_factor=progression_factor,
        )

        super().__init__(
            inner, lr, batch_size, lr_patience, lr_factor, lr_minimum, loss_weights
        )

        self.initial_filters = initial_filters
        self.complexity_factor = complexity_factor
        self.progression_factor = progression_factor
        self.include_eog = include_eog
        self.num_channels = num_channels

    def channels_prediction_EEGONLY(self, x_eegs):
        eegshape = x_eegs.shape

        num_eegs = eegshape[1]

        signal_len = eegshape[2]
        num_epochs = int(signal_len / 128 / 30)

        votes = torch.zeros(num_epochs, 5)  # fordi vi summerer løbende

        for i in range(num_eegs):
            x_eeg = x_eegs[:, i, ...]

            x_eeg = torch.unsqueeze(x_eeg, 1)

            pred = self(x_eeg)
            pred = torch.nn.functional.softmax(pred, dim=1)
            pred = torch.squeeze(pred)
            pred = pred.swapaxes(0, 1)
            pred = pred.cpu()

            votes = torch.add(votes, pred)

        votes = torch.argmax(votes, axis=1)

        return votes

    def channels_prediction(self, x_eegs, x_eogs):
        eegshape = x_eegs.shape
        eogshape = x_eogs.shape

        num_eegs = eegshape[1]
        num_eogs = eogshape[1]

        assert eegshape[2] == eogshape[2]

        signal_len = eegshape[2]
        num_epochs = int(signal_len / 128 / 30)

        votes = torch.zeros(num_epochs, 5)  # fordi vi summerer løbende

        for i in range(num_eegs):
            for p in range(num_eogs):
                x_eeg = x_eegs[:, i, ...]
                x_eog = x_eogs[:, p, ...]

                x_eeg = torch.unsqueeze(x_eeg, 1)
                x_eog = torch.unsqueeze(x_eog, 1)

                x_temp = torch.cat([x_eeg, x_eog], dim=1)

                pred = self(x_temp)
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = torch.squeeze(pred)
                pred = pred.swapaxes(0, 1)
                pred = pred.cpu()

                votes = torch.add(votes, pred)

        votes = torch.argmax(votes, axis=1)

        return votes

    def training_step(self, batch, _):
        if self.include_eog is True:
            x_eeg, x_eog, ybatch, _ = batch

            assert len(x_eog.shape) == 3
            assert x_eog.shape[1] == 1
            xbatch = torch.cat((x_eeg, x_eog), dim=1)
        else:
            x_eeg, ybatch, _ = batch
            xbatch = x_eeg

        assert len(x_eeg.shape) == 3
        assert x_eeg.shape[1] == 1
        assert len(ybatch.shape) == 2

        pred = self(xbatch)

        step_loss, _, _, _ = self.compute_train_metrics(pred, ybatch)

        self.training_step_outputs.append(step_loss)

        return step_loss

    def validation_step(self, batch, _):
        """Step per record"""

        # make sure  a single record was passed (no batching):
        assert batch[0].shape[0] == 1

        if self.include_eog is True:
            x_eeg, x_eog, ybatch, _ = batch

            assert len(x_eog.shape) == 3
            assert x_eog.shape[1] == 1
            xbatch = torch.cat((x_eeg, x_eog), dim=1)
        else:
            x_eeg, ybatch, _ = batch
            xbatch = x_eeg

        assert len(x_eeg.shape) == 3
        assert x_eeg.shape[1] == 1
        assert len(ybatch.shape) == 2

        pred = self(xbatch)

        step_loss, step_acc, step_kap, step_f1 = self.compute_train_metrics(
            pred, ybatch
        )

        assert (
            (step_acc is not None) and (step_kap is not None) and (step_f1 is not None)
        )

        # detach metrics from graph and move to cpu:
        step_loss = step_loss.cpu().detach()
        step_acc = step_acc.cpu().detach()
        step_kap = step_kap.cpu().detach()
        step_f1 = step_f1.cpu().detach()
        pred = pred.cpu().detach()
        ybatch = ybatch.cpu().detach()

        self.validation_step_loss.append(step_loss)
        self.validation_step_acc.append(step_acc)
        self.validation_step_kap.append(step_kap)
        self.validation_step_f1.append(step_f1)

        pred = torch.swapdims(pred, 1, 2)
        pred = torch.reshape(pred, (-1, 5))
        pred = torch.argmax(pred, dim=1)
        ybatch = torch.flatten(ybatch)

        self.validation_labels.append(ybatch)
        self.validation_preds.append(pred)

    def test_step(self, batch, _):
        """Step per record"""

        if self.include_eog is True:
            x_eeg, x_eog, ybatch, meta = batch

            assert len(x_eog.shape) == 3
            assert len(x_eeg.shape) == 3
            channels_pred = self.channels_prediction(x_eeg, x_eog)
        else:
            x_eeg, ybatch, meta = batch
            assert len(x_eeg.shape) == 3
            channels_pred = self.channels_prediction_EEGONLY(x_eeg)

        ybatch = torch.flatten(ybatch)

        log_test_step(
            "results",
            self.logger.version,
            dataset=meta["dataset"][0],
            subject=meta["subject"][0],
            record=meta["record"][0],
            channel_pred=channels_pred,
            labels=ybatch,
        )
