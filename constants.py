from enum import Enum
import torchaudio

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

SAMPLING_RATE = 16000
SAMPLE_COUNT = 16000
N_FFT = 1024
HOP_LENGTH=512
N_MELS = 64

HOW_MANY_EPOCHS_WAIT_FOR_IMPROVEMENT = 10

TRANSFORMATION = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLING_RATE,  # The sampling rate of the audio input
    n_fft=N_FFT,                # The size of the FFT(Fast Fourier Transform) - so basically resolution of frequency axis
    hop_length=HOP_LENGTH,      # The number of audio samples between frames - step between fft windows - "resolution" of time axis
    n_mels=N_MELS               # The number of mel filterbanks - how many frequency bins will spectrogram have
)

class DatasetGroup(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2