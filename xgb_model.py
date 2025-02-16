from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder

import torchaudio
from dataset_wrapper import CommandsTrainDataset
from collections.abc import Sequence

from skopt import BayesSearchCV
from skopt.space import Real, Integer

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class XGBDataWrapper(Sequence):
    """
    This class is a wrapper which simulates
    accessing a specific index of a pytorch DataSet like a list.
    """

    def __init__(self, data, constantIndex, sendToGPU: False) -> None:
        self._data = data
        self._constantIndex = constantIndex
        self._sendToGPU

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):
        if self._sendToGPU:
            tensor = self._data[index][self._constantIndex]
            return tensor.detach().cpu().numpy()
        return self._data[index][self._constantIndex]


def getTrainingData() -> tuple:
    transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
    )

    pytorchDataset = CommandsTrainDataset(DEVICE, 16000, 16000, transformation)
    trainingData = XGBDataWrapper(pytorchDataset, 0, True if DEVICE == "cuda"
                                  else False)
    trainingLabels = XGBDataWrapper(pytorchDataset, 1)

    return trainingData, trainingLabels


def createPipeline() -> Pipeline:
    steps = [
        ('targetEncoder', TargetEncoder())
    ]
    if DEVICE == "cuda":
        steps.append(('model', XGBClassifier(random_state=7,
                                             tree_method='gpu_hist',
                                             predictor='gpu_predictor',
                                             gpu_id=0)))
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

    return BayesSearchCV(pipeline, hyperParametersLimits, cv=2, n_iter=1,
                         scoring='roc_auc', random_state=7)


def main() -> None:
    trainingData, trainingLabels = getTrainingData()

    pipe = createPipeline()

    search = createHyperparameterOptimization(pipe)

    print(DEVICE)

    search.fit(trainingData, trainingLabels)

    print("Best estimator")
    print(search.best_estimator_)
    print("")

    print("Best score")
    print(search.best_score_)
    print("")

    print("Score")
    print(search.score(trainingData, trainingLabels))
    print("")


if __name__ == "__main__":
    main()
