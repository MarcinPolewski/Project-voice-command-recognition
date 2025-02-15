import torchaudio
import torch
from cnn2d_convoluted_network import CNN_2d
from dataset_wrapper import CommandsTrainDataset
from constants import CLASS_MAPPINGS


def predict(model, input, target):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = CLASS_MAPPINGS[predicted_index]
        expected = CLASS_MAPPINGS[target]

    return predicted, expected


if __name__ == "__main__":
    # load back the model
    model = CNN_2d()
    state_dict = torch.load("backup/cnn_2d_{epoch}")
    model.load_state_dict(state_dict)

    target_sampling_rate = 16000
    target_number_of_samples = 16000
    transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sampling_rate, n_fft=1024, hop_length=512, n_mels=64
    )

    validation_data = CommandsTrainDataset(
        device="cpu",
        target_sampling_rate=target_sampling_rate,
        target_number_of_samples=target_number_of_samples,
        transformation=transformation,
    )

    input, target = validation_data[0]
    print(input.shape)
    print(model(input))
