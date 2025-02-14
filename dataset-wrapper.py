from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class CommandsDataset(Dataset):
    def __init__(
        self,
        path_to_paths,  # path to txt file fith paths of audio files for test/train/validation
    ):
        self.data = self._load_data_from_file(path_to_paths)

        pass

    def __len__(self):
        pass

    def __getitem__(self, item_idx):
        pass

    def _preproces_sample(self, sample):
        pass

    def _load_data_from_file(self, path_to_paths):
        with open(path_to_paths, "r") as file:
            for line in file:
                commandClass = line.split("/")[0]
                path = base_path + line


class CommandsTrainDataset(CommandsDataset):
    def __init__(self):
        super().__init__("./train/")


class CommandsTestDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item_idx):
        pass

    def _preproces_sample(self, sample):
        pass
