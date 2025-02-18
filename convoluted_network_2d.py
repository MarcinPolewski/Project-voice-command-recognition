from torch import nn
from torch.utils.data import DataLoader
import torch
import torchaudio
from tqdm import tqdm


import constants
from dataset_wrapper import CommandsTrainDataset, CommandsTestDataset, CommandsValidateDataset


class CNN_2d_Trainer:
    def __init__(self, device):
        self.device = device

    def _create_data_loader(self, dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size)

    def train(self, model, dataset, loss_function, optimiser, epochs, batch_size):
        data_loader = self._create_data_loader(dataset, batch_size)

        for epoch in range(epochs):
            print(f"training {epoch} epoch")

            for input, target_output in tqdm(data_loader):
                # @TODO niepotrzebne przypisanie - juz w datasecie jest
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

class CNN_2d_Tester:

    @staticmethod
    def predict(model, input, target):
        model.eval()
        with torch.no_grad():
            predictions = model(input)
            predicted_index = predictions[0].argmax(0)
            predicted = constants.CLASS_MAPPINGS[predicted_index]
            expected = constants.CLASS_MAPPINGS[target]

        return predicted, expected
    
    @staticmethod
    def _test_base_method(model, device, dataset):        
        data_loader = DataLoader(dataset=dataset, batch_size=1)

        correct_predictions_count = 0

        for model_input, expected_output in tqdm(data_loader):
            predicted_class, expected_class = CNN_2d_Tester.predict(model, input, expected_output)
            correct_predictions_count+=1
            print(predicted_class, expected_class)

        score = correct_predictions_count / len(dataset)
        return score

    @staticmethod
    def test_model(model, device):
            
        target_sampling_rate = constants.SAMPLING_RATE
        target_number_of_samples = constants.SAMPLE_COUNT
        transformation = constants.TRANSFORMATION
        
        dataset = CommandsTestDataset(device, target_sampling_rate, target_number_of_samples, transformation)
        return CNN_2d_Tester._test_base_method(model, device, dataset)

    @staticmethod
    def validate_model(model, device):
        target_sampling_rate = constants.SAMPLING_RATE
        target_number_of_samples = constants.SAMPLE_COUNT
        transformation = constants.TRANSFORMATION

        dataset = CommandsValidateDataset(device, target_sampling_rate, target_number_of_samples, transformation)
        return CNN_2d_Tester._test_base_method(model, device, dataset)

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


def ex_main():
    print(str(torchaudio.list_audio_backends()))



    # dataset = CommandsTrainDataset(
    #     device, target_sampling_rate, target_number_of_samples, transformation
    # )


def train():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.mps.is_available():
    #     device = "mps"

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
def test():
    # set device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.mps.is_available():
    #     device = "mps"

    # load model
    model = CNN_2d()
    state_dict = torch.load("backup/cnn_2d_9")
    model.load_state_dict(state_dict)

    score = CNN_2d_Tester.test_model(model, device)
    print(f"accuracy: {score}")

def main():
    train()
    test()

if __name__ == "__main__":
    main()
