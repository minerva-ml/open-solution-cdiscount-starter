import numpy as np


class PredictionAverage():
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y, validation_data=None, img_dataset_filepath=None):
        for name, step in self.steps:
            step.fit(X, y, validation_data=validation_data, img_dataset_filepath=img_dataset_filepath)
        return self

    def predict(self, X, img_dataset_filepath=None):
        avg_pred_proba = self.predict_proba(X, img_dataset_filepath)
        return np.argmax(avg_pred_proba, axis=1)

    def predict_proba(self, X, img_dataset_filepath=None):
        predictions = np.stack([step.predict(X, img_dataset_filepath) for name, step in self.steps])
        avg_pred = predictions.mean(axis=0)
        return avg_pred