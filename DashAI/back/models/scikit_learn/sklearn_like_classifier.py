import numpy as np

from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset
from DashAI.back.models.scikit_learn.sklearn_like_model import SklearnLikeModel


class SklearnLikeClassifier(SklearnLikeModel):
    """Class for handling sklearn-like classifier models."""

    def predict(self, x_pred: DashAIDataset) -> np.ndarray:
        """Make a prediction with the model.

        Parameters
        ----------
        x_pred : DashAIDataset
            Dataset with the input data columns.

        Returns
        -------
        np.ndarray
            Array with the predicted target values for x_pred
        """
        if isinstance(x_pred, DashAIDataset):
            x_pred = x_pred.to_pandas()
        return super().predict_proba(x_pred)
