from sklearn.manifold import SpectralEmbedding as SpectralEmbeddingOperation

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


class SpectralEmbeddingSchema(BaseSchema):
    n_components: schema_field(
        int_field(ge=1),
        2,
        "The dimension of the projection subspace.",
    )  # type: ignore
    affinity: schema_field(
        enum_field(
            ["nearest_neighbors", "rbf", "precomputed", "precomputed_nearest_neighbors"]
        ),  # {'nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'} or callable
        "nearest_neighbors",
        "The affinity matrix to use for the computation of the embedding.",
    )  # type: ignore
    gamma: schema_field(
        none_type(float_field(gt=0)),
        None,
        "Kernel coefficient for rbf and nearest_neighbors affinities.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "Determines the random number generator used to initialize the centers. Pass an int for reproducible output across multiple function calls.",
    )  # type: ignore
    eigen_solver: schema_field(
        enum_field(["arpack", "lobpcg", "amg"]),
        None,
        "The eigenvalue decomposition strategy to use.",
    )  # type: ignore
    eigen_tol: schema_field(
        union_type(float_field(gt=0), enum_field(["auto"])),
        "auto",
        "Tolerance for 'arpack' or 'lobpcg'.",
    )  # type: ignore
    n_neighbors: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Number of nearest neighbors for nearest_neighbors affinity.",
    )  # type: ignore
    n_jobs: schema_field(
        none_type(int_field()),
        None,
        "The number of parallel jobs to run.",
    )  # type: ignore


class SpectralEmbedding(SklearnWrapper, SpectralEmbeddingOperation):
    """Scikit-learn's SpectralEmbedding wrapper for DashAI."""

    SCHEMA = SpectralEmbeddingSchema
    DESCRIPTION = "Spectral embedding for non-linear dimensionality reduction."
