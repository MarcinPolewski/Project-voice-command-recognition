from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline

import torchaudio
from dataset_wrapper import CommandsTrainDataset, CommandsTestDataset
from collections.abc import Sequence

from skopt import BayesSearchCV
from skopt.space import Real, Integer

import numpy as np

import torch

import time


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if DEVICE == "cuda":
    import cupy as cp


class XGBDataWrapper(Sequence):
    """
    This class is a wrapper which simulates
    accessing a specific index of a pytorch DataSet like a list.
    """

    def __init__(self, data, constantIndex, sendToGPU=False) -> None:
        self._data = data
        self._constantIndex = constantIndex
        self._sendToGPU = sendToGPU

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):
        tensor = self._data[index][self._constantIndex]
        if self._sendToGPU:
            tensor = tensor.flatten()
            return tensor.detach().cpu().numpy()
        return self._data[index][self._constantIndex]


def getData(DatasetClass) -> tuple:
    transformation = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)

    # MelSpectrogram(
    #    sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
    # )

    pytorchDataset = DatasetClass(DEVICE, 16000, 16000, transformation)
    trainingData = XGBDataWrapper(pytorchDataset, 0, True if DEVICE == "cuda"
                                  else False)
    trainingLabels = XGBDataWrapper(pytorchDataset, 1)

    trainingData = np.array([item.flatten() for item in trainingData])
    trainingLabels = np.array([item for item in trainingLabels])

    if DEVICE == "cuda":
        trainingData = cp.asarray(trainingData)

    return trainingData, trainingLabels


def createPipeline() -> Pipeline:
    steps = []
    if DEVICE == "cuda":
        steps.append(('model', XGBClassifier(random_state=7,
                                             device="cuda:0")))
    else:
        steps.append(('model', XGBClassifier(random_state=7)))
    return Pipeline(steps=steps)


def createHyperparameterOptimization(pipeline):
    hyperParametersLimits = {
        'model__max_depth': Integer(2, 8),
        'model__learning_rate': Real(0.0001, 1.0, prior='log-uniform'),
        'model__subsample': Real(0.5, 1.0),
        'model__colsample_bytree': Real(0.5, 1.0),
        'model__colsample_bylevel': Real(0.5, 1.0),
        'model__colsample_bynode': Real(0.5, 1.0),
        'model__reg_alpha': Real(0.0, 10.0),
        'model__reg_lambda': Real(0.0, 10.0),
        'model__gamma': Real(0.0, 10.0)
    }

    return BayesSearchCV(pipeline, hyperParametersLimits, cv=2, n_iter=100,
                         scoring='accuracy', random_state=7)


def main() -> None:
    print("Loading training data...")
    startTime = time.perf_counter()
    trainingData, trainingLabels = getData(CommandsTrainDataset)
    endTime = time.perf_counter()
    print("Loaded training data")
    print(f"Loading time: {endTime - startTime}")

    # pipe = createPipeline()

    # search = createHyperparameterOptimization(pipe)

    # print(f"Running on {DEVICE}")

    # search.fit(trainingData, trainingLabels)

    # print("Best estimator")
    # print(search.best_estimator_)
    # print("")

    # print("Best score")
    # print(search.best_score_)
    # print("")

    # print("Score")
    # print(search.score(trainingData, trainingLabels))
    # print("")

    # print("Predictions and data")
    # print(search.predict(trainingData))
    # print(trainingLabels)

    # model = XGBClassifier()
    # model.fit(trainingData, trainingLabels)
    # model.score()

    # print("")
    # print("Saving model")
    # model = search.best_estimator_.steps[0][1]
    # model.save_model("./xgboost_model_backup.json")

    startTime = time.perf_counter()
    model = XGBClassifier()
    model.fit(trainingData, trainingLabels)
    endTime = time.perf_counter()

    print(f"Training time: {endTime - startTime}")
    print("")

    print("Score on training data")
    print(model.score(trainingData, trainingLabels))
    print("")

    print("Loading test data")
    startTime = time.perf_counter()
    testData, testLabels = getData(CommandsTestDataset)
    endTime = time.perf_counter()
    print("Test data loaded")
    print(f"Loading time: {endTime - startTime}")

    # model = XGBClassifier()
    # model.load_model("./xgboost_model_backup.json")

    print("")
    print(f"Model score: {model.score(testData, testLabels)}")
    print("")

    print("Model prediction, data, search prediction")
    print(model.predict(testData))

    print(testLabels)

    # print(search.predict(testData))


if __name__ == "__main__":
    main()
