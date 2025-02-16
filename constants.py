from enum import Enum

TRAIN_LIST_PATH = "./train/train_list.txt"
VALIDATE_LIST_PATH = "./train/validation_list.txt"
TESTING_LIST_PATH = "./train/testing_list.txt"
AUDIO_BASE_PATH = "./train/audio/"

CLASS_MAPPINGS = [
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "four",
    "go",
    "happy",
    "house",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "wow",
    "yes",
    "zero",
]

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1 

validation_percentage = 10.0
test_percentage = 10.0

class DatasetGroup(Enum):
    TRAIN = 0
    VALIDATE = 1 
    TEST = 2