from torch import nn
from torch.utils.data import DataLoader
import torch
import torchaudio
from torchsummary import summary

import constants


class CNN_1d(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=4,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=4,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=4,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=2,
                padding=4,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        # neural network
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 4, len(constants.CLASS_MAPPINGS))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    model = CNN_1d()
    summary(model, input_size=(1, 16000))
    