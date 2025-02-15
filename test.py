import torchaudio
import torch
from convoluted_network_2d import CNN_2d
from convoluted_network_1d import CNN_1d
from dataset_wrapper import CommandsTestDataset
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
    model = CNN_1d()
    state_dict = torch.load("backup/cnn_1d_9")
    model.load_state_dict(state_dict)

    target_sampling_rate = 16000
    target_number_of_samples = 16000
    transformation = None

    testing_data = CommandsTestDataset(
        device="cpu",
        target_sampling_rate=target_sampling_rate,
        target_number_of_samples=target_number_of_samples,
        transformation=transformation,
    )

    # input, target = validation_data[
    #     0
    # ]  # here input has 3 dimensions -> [number_of_chanels, frequency_axis, time_axis]
    # input.unsqueeze_(
    #     0
    # )  # addin another dimension - model expexts [batch_size, number_of_chanels, ...]
    # # 0 is the index, where we add new dimension

    ct = 0
    for input, target in testing_data:
        input.unsqueeze_(0)

        predited, expected = predict(model, input, target)
        print(f'predicted: {predited}, expected: {expected}')
        ct += predited == expected

    print(f'Accuraccy: {ct / len(testing_data)}')
