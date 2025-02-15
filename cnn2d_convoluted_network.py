from torch import nn
from torch.utils.data import DataLoader
import torch
import torchaudio
import torchsummary

import constants
from dataset_wrapper import CommandsTrainDataset


class CNN_2d_Trainer:
    def __init__(self, device):
        self.device = device

    def _create_data_loader(self, dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size)

    def train(self, model, dataset, loss_function, optimiser, epochs, batch_size):
        data_loader = self._create_data_loader(dataset, batch_size)

        for epoch in range(epochs):
            print(f"training {epoch} epoch")

            for input, target_output in data_loader:
                input = input.to(self.device)
                target_output = target_output.to(self.device)

                # loss
                model_output = model(input)
                loss = loss_function(model_output, target_output)

                # backpropagation
                optimiser.zero_grad()  # reset gradient values
                loss.backward()
                optimiser.step()

            # save model
            torch.save(model.state_dict(), f"backup/cnn_2d_{epoch}")


class CNN_2d(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolution_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.convolution_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.convolution_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.convolution_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # neural network:
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 3, len(constants.CLASS_MAPPINGS))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.convolution_1(input_data)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        x = self.convolution_4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


def main():
    print(str(torchaudio.list_audio_backends()))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # ===== constants and declarations ====
    trainer = CNN_2d_Trainer(device)
    model = CNN_2d()
    target_sampling_rate = 16000
    target_number_of_samples = 16000
    transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sampling_rate, n_fft=1024, hop_length=512, n_mels=64
    )
    learning_rate = 0.01
    epochs = 10
    batch_size = 32
    loss_function = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # dataset = CommandsTrainDataset(
    #     device, target_sampling_rate, target_number_of_samples, transformation
    # )
    dataset = CommandsTrainDataset(
        device=device,
        target_sampling_rate=target_sampling_rate,
        target_number_of_samples=target_number_of_samples,
        transformation=transformation,
    )

    # === actual training
    trainer.train(
        model=model,
        dataset=dataset,
        loss_function=loss_function,
        optimiser=optimiser,
        epochs=epochs,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
