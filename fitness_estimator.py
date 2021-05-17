from typing import Dict

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset

from main import MCSettings


DEVICE = "cuda" if th.cuda.is_available() else "cpu"


class GenericDataset(Dataset):
    """
    A generic dataset class
    """

    def __init__(self,
                 xs: th.tensor,
                 ys: th.tensor):
        super(Dataset, self).__init__()
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


class ArtifactsBuffer:
    def __init__(self,
                 settings: MCSettings):
        self.capacity = settings.buffer_capacity
        self.artifacts = []
        self.fitnesses = []

    def add(self,
            artifact: np.ndarray,
            fitness: float) -> None:
        if len(artifact) == self.capacity:
            n = np.random.randint(low=0,
                                  high=self.capacity)
            self.artifacts.pop(n)
            self.fitnesses.pop(n)
        self.artifacts.append(artifact)
        self.fitnesses.append(fitness)

    @staticmethod
    def _prepare_dataloaders(xs: th.Tensor,
                             ys: th.Tensor,
                             batch_size: int) -> Dict[str, th.utils.data.DataLoader]:
        """
        Prepare the dataloaders for the desired task.

        :param xs: The array of input samples
        :param ys: The array of labels
        :param batch_size: The batch size
        :return: A dictionary with both training and testing dataloaders
        """
        dataset = GenericDataset(xs=xs,
                                 ys=ys)
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = th.utils.data.random_split(dataset=dataset,
                                                                 lengths=[train_size, test_size],
                                                                 generator=th.Generator())
        train_dataset_loader = th.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
        test_dataset_loader = th.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
        return {
            "train": train_dataset_loader,
            "test": test_dataset_loader,
        }

    def prepare(self) -> Dict[str, th.utils.data.DataLoader]:
        n_valid_artifacts = np.sum(self.fitnesses)
        if n_valid_artifacts <= len(self.artifacts) / 2:
            diff = len(self.artifacts) - (2 * n_valid_artifacts)
            artifacts = []
            fitnesses = []
            # oversampling filtered in artifacts by rotation around Y
            high_performing_idxs = np.where(np.asarray(fitnesses) == 1.)
            for j in high_performing_idxs[0]:
                for r in range(1, 4):
                    artifacts.append(np.rot90(m=artifacts[j],
                                              k=r,
                                              axes=(0, 2)))
                    fitnesses.append(1.)
                    diff -= 1
                if diff <= 0:
                    break
            # undersampling if the dataset would still be unbalanced
            if diff > 0:
                removable_idxs = np.where(np.asarray(fitnesses) == 0)
                keep_idxs = np.append(high_performing_idxs[0],
                                      removable_idxs[0][0:len(removable_idxs[0]) - int(diff) + 1])
                artifacts = artifacts[keep_idxs]
                fitnesses = fitnesses[keep_idxs]
        else:
            artifacts = self.artifacts
            fitnesses = self.fitnesses
        artifacts = th.as_tensor(artifacts)
        fitnesses = th.as_tensor(fitnesses)
        # pass to dataset builder, return dataloaders
        return self._prepare_dataloaders(
            xs=artifacts,
            ys=fitnesses,
            batch_size=2)


class ConvolutionalBlock(nn.Module):
    def __init__(self, **kwargs):
        super(ConvolutionalBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(
                in_channels=kwargs.get('in_channels_conv'),
                out_channels=kwargs.get('out_channels_conv'),
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm3d(
                num_features=kwargs.get('out_channels_conv')
            ),
            nn.ReLU()
        ).to(DEVICE)

    def forward(self, x):
        return self.seq(x)


class ResidualBlock(nn.Module):
    def __init__(self, **kwargs):
        super(ResidualBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(
                in_channels=kwargs.get('in_out_channels_res'),
                out_channels=kwargs.get('in_out_channels_res'),
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm3d(
                num_features=kwargs.get('in_out_channels_res')
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=kwargs.get('in_out_channels_res'),
                out_channels=kwargs.get('in_out_channels_res'),
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm3d(
                num_features=kwargs.get('in_out_channels_res')
            )
        ).to(DEVICE)

    def forward(self, x):
        residual = x
        x = self.seq(x)
        x += residual
        return nn.ReLU()(x)


class FitnessEstimator(nn.Module):
    def __init__(self, **kwargs):
        super(FitnessEstimator, self).__init__()
        self.seq = nn.Sequential(
            ConvolutionalBlock(**kwargs),
            ResidualBlock(**kwargs),
            ResidualBlock(**kwargs),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=kwargs.get('n_features'),
                      out_features=1),
            nn.ReLU(),
            nn.Softmax(dim=1)
        ).to(DEVICE)

    def forward(self, x):
        return self.seq(x)


if __name__ == "__main__":
    from torchsummary import summary

    shape = (2, 5, 10, 5)
    conv_channels = 16
    shape_after_conv = (16, 5, 10, 5)

    args = {
        'in_channels_conv': shape[0],
        'out_channels_conv': conv_channels,
        'in_out_channels_res': conv_channels,
        'n_features': 16 * shape[1] * shape[2] * shape[3],
    }
    print('*** Neural networks blocks summaries: ***')
    print('\n\tConvolutionalBlock:')
    summary(ConvolutionalBlock(**args), shape)
    print('\n\tResidualBlock:')
    summary(ResidualBlock(**args), shape_after_conv)
    print('\n\tFitnessEstimator:')
    summary(FitnessEstimator(**args), shape)
