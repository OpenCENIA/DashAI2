from sklearn.kernel_approximation import RBFSampler as RBFSamplerOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    enum_field,
    float_field,
    int_field,
    none_type,
    schema_field,
    union_type,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class RBFSamplerSchema(BaseSchema):
    gamma: schema_field(
        union_type(enum_field(["scale"]), float_field(gt=0)),
        1.0,
        "Parameter of the RBF kernel.",
    )  # type: ignore
    n_components: schema_field(
        int_field(ge=1),
        100,
        "The number of features to construct.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "Pseudo-random number generator to control the generation of the random weights and random offset when fitting the training data. Pass an int for reproducible output across multiple function calls.",
    )  # type: ignore


class RBFSampler(SklearnWrapper, RBFSamplerOperation):
    """Scikit-learn's RBFSampler wrapper for DashAI."""

    SCHEMA = RBFSamplerSchema
    DESCRIPTION = "Approximates the feature map of an RBF kernel by Monte Carlo approximation of its Fourier transform."
