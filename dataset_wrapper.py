import os
from torch.utils.data import Dataset
import torchaudio
from train_list_generator import Train_List_Generator
import constants
import torch
import hashlib
import re


class CommandsDataset(Dataset):
    CLASS_IDX = 1
    SAMPLE_PATH_IDX = 0

    def __init__(
        self,
        dataset_group,  # path to txt file fith paths of audio files for test/train/validation - for instance ./train/validation_list.txt
        device,
        target_sampling_rate,  # all returns samples will have this sr
        target_number_of_samples,  # how many samples should target recording have
        transformation,
    ):
        self.device = device
        self.target_sampling_rate = target_sampling_rate
        self.target_number_of_samples = target_number_of_samples
        self.transformation = transformation
        self.validation_percentage = constants.validation_percentage
        self.test_percentage = constants.test_percentage

        Train_List_Generator.generate()
        self.dataset_group = dataset_group
        self.data = self._load_sound_list()

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

        # Check if the file exists
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"File not found: {sample_path}")
        elif not os.path.isfile(sample_path):
            raise FileNotFoundError(f"NOT A FILE: {sample_path}")

        sample, sampling_rate = torchaudio.load(sample_path)
        sample.to(self.device)

        sample = self._preproces_sample(sample, sampling_rate)
        if self.transformation:
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

        return sample

    def _preproces_sample(self, sample, sampling_rate):
        sample = self._resample(sample, sampling_rate)
        sample = self._mix_down(sample)
        sample = self._adjust_length(sample)
        return sample

    def _is_train_test_or_validate(self, filename):
        """
        assign this file to set in a way to reduce risk of train sample being resuded in testing etc
        here where files are assigned only depens on filename and percentage values provided as arguments
        also realted files(same command by the same actor will be assigned to the same set)
        """
        base_name = os.path.basename(filename)

        hash_name = re.sub(r'_nohash_.*$', '', base_name)

        hash_name_hashed = hashlib.sha1(hash_name.encode("utf-8")).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (constants.MAX_NUM_WAVS_PER_CLASS + 1)) *
                            (100.0 / constants.MAX_NUM_WAVS_PER_CLASS))
        
        if percentage_hash < self.validation_percentage:
            return constants.DatasetGroup.VALIDATE
        elif percentage_hash < (self.test_percentage + self.validation_percentage):
            return constants.DatasetGroup.TEST
        return constants.DatasetGroup.TRAIN

    # def _load_sound_list(self, path_to_paths):
    #     """
    #     loads a file with audio samples(for training/testing/validation) to python object
    #     """

    #     data = []

    #     with open(path_to_paths, "r") as file:
    #         base_path = constants.AUDIO_BASE_PATH
    #         for line in file:
    #             sample_class = line.split("/")[0]
    #             path = base_path + line
    #             path = path.strip()

    #             data_row = [path, sample_class]
    #             data.append(data_row)



    #     return data


    def _load_sound_list(self):

        data = []

        base_path = constants.AUDIO_BASE_PATH
        
        for class_name in constants.CLASS_MAPPINGS:
            class_folder_path = base_path + class_name
            all_files_of_this_class = os.listdir(class_folder_path)

            for filename in all_files_of_this_class: # file is just a file name
                if self.dataset_group == self._is_train_test_or_validate(filename):
                    full_filename = class_name + "/" + filename

                    result_path = base_path + full_filename
                    result_path.strip()

                    data_row = [result_path, class_name]
                    data.append(data_row)

        return data

class CommandsTrainDataset(CommandsDataset):
    def __init__(
        self,
        device,
        target_sampling_rate,  # all returns samples will have this sr
        target_number_of_samples,  # how many samples should target recording have
        transformation,
    ):
        super().__init__(
            constants.DatasetGroup.TRAIN,  
            device,
            target_sampling_rate,  # all returns samples will have this sr
            target_number_of_samples,  # how many samples should target recording have
            transformation,
        )


class CommandsTestDataset(CommandsDataset):
    def __init__(
        self,
        device,
        target_sampling_rate,  # all returns samples will have this sr
        target_number_of_samples,  # how many samples should target recording have
        transformation,
    ):
        super().__init__(
            constants.DatasetGroup.TEST, 
            device,
            target_sampling_rate,  # all returns samples will have this sr
            target_number_of_samples,  # how many samples should target recording have
            transformation,
        )

class CommandsValidateDataset(CommandsDataset):
    def __init__(
        self,
        device,
        target_sampling_rate,  # all returns samples will have this sr
        target_number_of_samples,  # how many samples should target recording have
        transformation,
    ):
        super().__init__(
            constants.DatasetGroup.VALIDATE,  
            device,
            target_sampling_rate,  # all returns samples will have this sr
            target_number_of_samples,  # how many samples should target recording have
            transformation,
        )

if __name__ == "__main__":
    trans  = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
    )
    ds = CommandsTestDataset("cpu", 16000, 16000, trans)
    print(len(ds))

    ds = CommandsTrainDataset("cpu", 16000, 16000, trans)
    print(len(ds))

    ds = CommandsValidateDataset("cpu", 16000, 16000, trans)
    print(len(ds))