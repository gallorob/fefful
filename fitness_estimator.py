import os
from typing import Dict, List
from operator import itemgetter

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from settings import MCSettings

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
        self.batch_size = settings.batch_size
        self.artifacts = []
        self.fitnesses = []

    @property
    def at_capacity(self):
        return len(self.fitnesses) == self.capacity

    def add(self,
            artifact: np.ndarray,
            fitness: float) -> None:
        # make space in buffer if needed
        if self.at_capacity:
            # TODO maybe only pop unbalancing examples?
            n = np.random.randint(low=0,
                                  high=self.capacity)
            self.artifacts.pop(n)
            self.fitnesses.pop(n)
        # add both artifact and fitness
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
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
        test_dataset_loader = th.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
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
            high_performing_idxs = np.where(np.asarray(self.fitnesses) == 1.)[0]
            for j in high_performing_idxs:
                for r in range(1, 4):
                    artifacts.append(np.rot90(m=self.artifacts[j],
                                              k=r,
                                              axes=(1, 3)))  # CxWxHxD, so rotate W and D around H
                    fitnesses.append(1.)
                    diff -= 1
                if diff <= 0:
                    break
            artifacts.extend(self.artifacts)
            fitnesses.extend(self.fitnesses)
            # undersampling if the dataset would still be unbalanced
            if diff > 0:
                removable_idxs = np.where(np.asarray(fitnesses) == 0.)[0]
                keep_idxs = np.append(high_performing_idxs,
                                      removable_idxs[0:len(removable_idxs) - int(diff) + 1])
                get_keepers = itemgetter(*keep_idxs)
                artifacts = get_keepers(artifacts)
                fitnesses = get_keepers(fitnesses)
        else:
            # TODO what if the data is unbalanced towards positive examples?
            artifacts = self.artifacts
            fitnesses = self.fitnesses
        artifacts = th.as_tensor(artifacts)
        fitnesses = th.as_tensor(fitnesses)
        # pass to dataset builder, return dataloaders
        return self._prepare_dataloaders(
            xs=artifacts,
            ys=fitnesses,
            batch_size=self.batch_size
        )


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
            nn.Dropout(p=0.6),
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
            nn.Dropout(p=0.6)
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
            nn.ReLU(),
            ResidualBlock(**kwargs),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=kwargs.get('n_features'),
                      out_features=1),
            nn.Sigmoid()
        ).to(DEVICE)

    def forward(self, x):
        return self.seq(x).squeeze()


class FitnessEstimatorWrapper:
    def __init__(self,
                 test_threshold: float,
                 net_args):
        self.net = FitnessEstimator(**net_args)
        self.test_threshold = test_threshold
        self.can_estimate = False
        self.writer = SummaryWriter()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.epoch = 0
        self.train_accuracy = 0.
        self.train_loss = 0.
        self.val_accuracy = 0.
        self.val_loss = 0.

    @staticmethod
    def _binary_acc(predictions,
                    labels):
        correct = (th.round(predictions) == labels).float().sum()
        return correct / len(labels)

    def train(self,
              dataloaders: Dict[str, th.utils.data.DataLoader],
              epochs: int):
        train_data = dataloaders.get('train')
        train_bs = train_data.batch_size
        test_data = dataloaders.get('test')
        # training
        train_loss = 0.
        train_acc = 0.
        for epoch in range(epochs):
            self.net.train()
            bar = tqdm(desc=f'Epoch {epoch + 1}',
                       total=len(train_data))
            for i, data in enumerate(train_data, 0):
                inputs, labels = data
                inputs, labels = inputs.float().to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # update metrics and display
                train_loss += loss.item()
                train_acc += self._binary_acc(predictions=outputs.cpu().detach(),
                                              labels=labels.cpu())
                bar.set_postfix_str(
                    s=f"Loss: {train_loss / train_bs}; Acc: {train_acc}")
                # log results at the end of training
                if i == len(train_data) - 1:
                    self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                    self.writer.add_scalar('Loss/train', train_loss / train_bs, epoch)
                    self.train_accuracy = train_acc
                    self.train_loss = train_loss / train_bs
                train_loss = 0.
                train_acc = 0.
                bar.update(n=1)
            bar.close()

            # testing
            correct = 0
            test_loss = 0.
            total = 0
            with th.no_grad():
                self.net.eval()
                bar = tqdm(desc=f'Filter test',
                           total=len(test_data))
                for j, data in enumerate(test_data):
                    sample, labels = data
                    sample, labels = sample.float().to(DEVICE), labels.to(DEVICE)
                    outputs = self.net(sample)
                    correct += self._binary_acc(predictions=outputs.cpu().detach(),
                                                labels=labels.cpu())
                    test_loss += self.criterion(outputs, labels) / labels.shape[0]
                    total += 1
                    bar.set_postfix_str(
                        s=f"Loss: {test_loss / total}; Acc: {correct / total}")
                    bar.update(n=1)
                bar.set_postfix_str(
                    s=f"Loss: {test_loss / total}; Acc: {correct / total}")
                bar.close()
                self.writer.add_scalar('Accuracy/test', correct / total, epoch)
                self.writer.add_scalar('Loss/test', test_loss / total, epoch)
                self.val_accuracy = correct / total
                self.val_loss = test_loss / total

            self.epoch += 1

        # check if the filter can be considered trained or not
        self.can_estimate = bool(self.test_threshold <= self.val_accuracy)

    def estimate(self,
                 artifacts: List[np.ndarray]) -> List[float]:
        """
        Estimate the fitness for a batch of artifacts.

        :param artifacts: List of N artifacts. Each artifact is a WxHxDxC NumPy array.
        :return: The tensor containing the estimated fitness (values between 0 and 1)
        """
        with th.no_grad():
            self.net.eval()
            artifacts = th.as_tensor(artifacts).float().to(DEVICE)
            return self.net(artifacts).squeeze().detach().cpu().numpy().tolist()

    def save(self,
             to_resume: bool,
             where: str):
        t = datetime.now()
        name = t.strftime('%Y%m%d%H%M%S')
        if to_resume:
            th.save({
                'epoch': self.epoch,
                'train_loss': self.train_loss,
                'train_accuracy': self.train_accuracy,
                'val_loss': self.val_loss,
                'val_accuracy': self.val_accuracy,
                'estimator_dict': self.net.state_dict(),
                'can_estimate': self.can_estimate,
                'optimizer': self.optimizer.state_dict()
            }, os.path.join(where, f'{name}.checkpoint'))
        else:
            th.save(self.net.state_dict(),
                    os.path.join(where, f'{name}_estimator.pth'))

    def load(self,
             to_resume: bool,
             where: str,
             timestep: str):
        if to_resume:
            checkpoint = th.load(os.path.join(where, f'{timestep}.checkpoint'))
            self.net.load_state_dict(checkpoint['estimator_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint.get('epoch', 0)
            self.can_estimate = checkpoint.get('can_estimate', False)
            self.train_loss = checkpoint.get('train_loss', 0)
            self.train_accuracy = checkpoint.get('train_accuracy', 0)
            self.val_loss = checkpoint.get('val_loss', 0)
            self.val_accuracy = checkpoint.get('val_accuracy', 0)
        else:
            self.net.load_state_dict(th.load(os.path.join(where, f'{timestep}_estimator.pth')))


if __name__ == "__main__":
    from torchsummary import summary

    shape = (2, 5, 7, 5)
    conv_channels = 16
    shape_after_conv = (16, 5, 7, 5)

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
