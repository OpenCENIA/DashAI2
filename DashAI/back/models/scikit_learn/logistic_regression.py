from typing import Optional

from sklearn.linear_model import LogisticRegression as _LogisticRegression

from DashAI.back.core.schema_fields import (
    BaseSchema,
    float_field,
    int_field,
    string_field,
)
from DashAI.back.models.scikit_learn.sklearn_like_model import SklearnLikeModel
from DashAI.back.models.tabular_classification_model import TabularClassificationModel


class LogisticRegressionSchema(BaseSchema):
    """Logistic Regression is a supervised classification method that uses a linear
    model plus a a logistic funcion to predict binary outcomes (it can be configured
    as multiclass via the one-vs-rest strategy).
    """

    penalty: Optional[
        string_field(
            description="Specify the norm of the penalty",
            default="l2",
            enum=["l2", "l1", "elasticnet"],
        )
    ]
    tol: float_field(
        description="Tolerance for stopping criteria.", default=0.0001, minimum=0
    )
    C: float_field(
        description="Inverse of regularization strength, smaller values specify "
        "stronger regularization. Must be a positive number.",
        default=1,
        minimum=0,
    )
    max_iter: int_field(
        description="Maximum number of iterations taken for the solvers to converge.",
        default=100,
        minimum=50,
    )


class LogisticRegression(
    TabularClassificationModel, SklearnLikeModel, _LogisticRegression
):
    """Scikit-learn's Logistic Regression wrapper for DashAI."""

    SCHEMA = LogisticRegressionSchema
