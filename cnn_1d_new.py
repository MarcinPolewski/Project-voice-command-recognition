from torch import nn
from torch.utils.data import DataLoader
import torch
import torchaudio
from torchsummary import summary
from constants import CLASS_MAPPINGS

import constants
from dataset_wrapper import CommandsTrainDataset, CommandsValidateDataset


def predict(model, device, inputs, targets):
    model.eval()
    inputs = inputs.to(device)
    with torch.no_grad():
        predictions = model(inputs)
        predicted_indices = predictions.argmax(1)
        predicted = [CLASS_MAPPINGS[idx] for idx in predicted_indices]
        expected = [CLASS_MAPPINGS[target] for target in targets]
    return predicted, expected


class CNN_1d(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # neural network
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.Dropout(p=0.2),
            nn.Linear(64, len(constants.CLASS_MAPPINGS)),
        )

    def forward(self, input_data):
        # input_data = input_data.to("mps")
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class CNN_1d_Trainer:
    def __init__(self, device):
        self.device = device

    def _create_data_loader(self, dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, model, train_dataset, validate_dataset, loss_function,
              optimiser, epochs, batch_size):

        train_data_loader = self._create_data_loader(
            train_dataset, batch_size)

        validate_data_loader = self._create_data_loader(
            validate_dataset, batch_size)

        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}", end=' ')

            # Enable training
            model.train()

            loss_train = 0.0
            for input, target in train_data_loader:
                # assign device
                input = input.to(self.device)
                target = target.to(self.device)

                # compute loss
                model_output = model(input)
                loss = loss_function(model_output, target)
                loss_train += loss.item()

                # update weights
                optimiser.zero_grad()  # reset gradient values
                loss.backward()
                optimiser.step()

            # Compute accuraccy and loss on train dataset
            ct_train = 0
            total_train = len(train_dataset)
            model.eval()
            for input, target in train_data_loader:
                input = input.to(self.device)
                target = target.to(self.device)

                predicted, expected = predict(
                    model, self.device, input, target)
                ct_train += sum(p == e for p, e in zip(predicted, expected))

            avg_loss_train = loss_train / total_train
            accuracy_train = ct_train / total_train

            # Compute accuraccy on validation dataset
            ct_validate = 0
            loss_validate = 0.0
            total_validate = len(validate_dataset)
            model.eval()
            with torch.no_grad():
                for input, target in validate_data_loader:
                    input = input.to(self.device)
                    target = target.to(self.device)

                    model_output = model(input)
                    loss = loss_function(model_output, target)
                    loss_validate += loss.item()

                    predicted, expected = predict(
                        model, self.device, input, target)
                    ct_validate += sum(p == e for p, e in zip(predicted, expected))

            avg_loss_validate = loss_validate / total_validate
            accuracy_validate = ct_validate / total_validate

            print(f"\rEpoch: {epoch}, train accu: {accuracy_train:.4f}, train loss: {avg_loss_train:.4f}, val accu: {accuracy_validate:.4f}, val loss: {avg_loss_validate:.4f}")

            # save model
            torch.save(model.state_dict(), f"backup/cnn_1d_{epoch}")


if __name__ == "__main__":
    # Configre backends
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    print(f'Assigned device: {device}')
    print(f'Available audio backends: {torchaudio.list_audio_backends()}')

    # Create model and assign device
    model = CNN_1d()
    model.to(device)

    # Display model architecture
    # summary(model, input_size=(1, 16000))

    # Create trainer
    trainer = CNN_1d_Trainer(device)

    # Configure model
    epochs = 25
    learning_rate = 0.0003
    batch_size = 512
    target_sampling_rate = 16000
    target_number_of_samples = 16000
    loss_function = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create training dataset
    train_dataset = CommandsTrainDataset(
        device=device,
        target_sampling_rate=target_sampling_rate,
        target_number_of_samples=target_number_of_samples,
        transformation=None,
    )

    # Create validating dataset
    validate_dataset = CommandsValidateDataset(
        device=device,
        target_sampling_rate=target_sampling_rate,
        target_number_of_samples=target_number_of_samples,
        transformation=None,
    )

    # Train model
    trainer.train(
        model=model,
        train_dataset=train_dataset,
        validate_dataset=validate_dataset,
        loss_function=loss_function,
        optimiser=optimiser,
        epochs=epochs,
        batch_size=batch_size,
    )
