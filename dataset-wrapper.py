from torch.utils.data import Dataset
import torchaudio
from train_list_generator import Train_List_Generator
import constants
import torch


class CommandsDataset(Dataset):
    CLASS_IDX = 1
    SAMPLE_PATH_IDX = 0

    def __init__(
        self,
        path_to_paths,  # path to txt file fith paths of audio files for test/train/validation - for instance ./train/validation_list.txt
        device,
        target_sampling_rate,  # all returns samples will have this sr
        target_number_of_samples,  # how many samples should target recording have
        transformation,
    ):
        self.device = device
        self.target_sampling_rate = target_sampling_rate
        self.target_number_of_samples = target_number_of_samples
        self.transformation = transformation
        Train_List_Generator.generate()
        self.data = self._load_sound_list(path_to_paths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        sound_sample, sampling_rate = self._load_sample(item_idx)
        label = self._get_label(item_idx)
        return sound_sample, label

    def _get_label(self, sample_idx):
        label_string = self.data[sample_idx][CommandsDataset.CLASS_IDX]
        label_digit = constants.CLASS_MAPPINGS.index(
            label_string
        )  # throws error when not found
        return label_digit

    def _load_sample(self, sample_idx):
        sample_path = self.data[sample_idx][CommandsDataset.SAMPLE_PATH_IDX]
        sample, sampling_rate = torchaudio.load(sample_path)
        sample.to(self.device)

        sample = self._preproces_sample(sample, sampling_rate)
        sample = self.transformation(sample)
        return sample, sampling_rate

    def _resample(self, sample, sampling_rate):
        """
        adjust sample to have the same sampling_rate as other samples
        """

        if sampling_rate != self.target_sampling_rate:
            resampler = torchaudio.transforms.Resample(
                sampling_rate, self.target_sampling_rate
            )
            sample = resampler(sample)
        return sample

    def _mix_down(self, sample):
        """
        "compress" sample to one audio chanel
        """

        if sample.shape[0] != 1:
            sample = torch.mean(sample, dim=0, keepdim=True)
        return sample

    def _adjust_length(self, sample):
        """
        make sure all samples have the same length
        sample - pytorch tensor of dimension (number_of_chanels, number_of_samples)
        """

        number_of_samples = sample.shape[1]

        # case 1 - we have not enough samples - fill the right side of "array" with 0
        if number_of_samples < self.target_number_of_samples:
            how_many_to_add = self.target_number_of_samples - number_of_samples
            padding = (0, how_many_to_add)
            sample = torch.nn.functional.pad(sample, padding)

        # case 2 - we have too many samples
        elif number_of_samples > self.target_number_of_samples:
            sample = sample[
                :, : self.target_number_of_samples
            ]  # truncate excess at the end

    def _preproces_sample(self, sample, sampling_rate):
        sample = self._resample(sample, sampling_rate)
        sample = self._mix_down(sample)
        sample = self._adjust_length(sample)
        return sample

    def _load_sound_list(self, path_to_paths):
        """
        loads a file with audio samples(for training/testing/validation) to python object
        """

        data = []

        with open(path_to_paths, "r") as file:
            base_path = constants.AUDIO_BASE_PATH
            for line in file:
                sample_class = line.split("/")[0]
                path = base_path + line

                data_row = [path, sample_class]
                data.append(data_row)

        return data
