from sklearn.manifold import TSNE as TSNEOperation

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


class TSNESchema(BaseSchema):
    n_components: schema_field(
        int_field(ge=1),
        2,
        "Dimension of the embedded space.",
    )  # type: ignore
    perplexity: schema_field(
        float_field(gt=0),
        30.0,
        "The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50.",
    )  # type: ignore
    early_exaggeration: schema_field(
        float_field(gt=0),
        12.0,
        "Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them.",
    )  # type: ignore
    learning_rate: schema_field(
        union_type(float_field(gt=0), enum_field(["auto"])),
        "auto",
        "The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high, the data may look like a 'ball' with any point approximately equidistant from its nearest neighbors.",
    )  # type: ignore
    n_iter: schema_field(
        int_field(ge=1),
        1000,
        "Maximum number of iterations for the optimization. Should be at least 250.",
    )  # type: ignore
    n_iter_without_progress: schema_field(
        int_field(ge=1),
        300,
        "Maximum number of iterations without progress before we abort the optimization, used after 250 initial iterations with early exaggeration.",
    )  # type: ignore
    min_grad_norm: schema_field(
        float_field(gt=0),
        1e-7,
        "If the gradient norm is below this threshold, the optimization will be stopped.",
    )  # type: ignore
    metric: schema_field(
        enum_field(
            [
                "cityblock",
                "cosine",
                "euclidean",
                "l1",
                "l2",
                "manhattan",
                "braycurtis",
                "canberra",
                "chebyshev",
                "correlation",
                "dice",
                "hamming",
                "jaccard",
                "kulsinski",
                "mahalanobis",
                "minkowski",
                "rogerstanimoto",
                "russellrao",
                "seuclidean",
                "sokalmichener",
                "sokalsneath",
                "sqeuclidean",
                "yule",
                "precomputed",
            ]
        ),  # str or callable
        "euclidean",
        "The metric to use when calculating distance between instances in a feature array.",
    )  # type: ignore
    # metric_params: schema_field(
    #     none_type(string_field()), # dict
    #     None,
    #     "Additional keyword arguments for the metric function.",
    # )  # type: ignore
    init: schema_field(
        enum_field(
            ["random", "pca"]
        ),  # {'random', 'pca'} or ndarray of shape (n_samples, n_components)
        "pca",
        "Initialization of embedding.",
    )  # type: ignore
    verbose: schema_field(
        int_field(ge=0),
        0,
        "Verbosity level.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "Determines the random number generator used for initialization.",
    )  # type: ignore
    method: schema_field(
        enum_field(["barnes_hut", "exact"]),
        "barnes_hut",
        "By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time. method='exact' will run on the slower, but exact, algorithm in O(N^2) time.",
    )  # type: ignore
    angle: schema_field(
        float_field(gt=0, le=1),
        0.5,
        "Only used if method='barnes_hut'. This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.",
    )  # type: ignore
    n_jobs: schema_field(
        none_type(int_field()),
        None,
        "The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.",
    )  # type: ignore
    # Deprecated since version 1.1
    # square_distances: schema_field(
    #     bool_field(),
    #     False,
    #     "If True, the input similarities are treated as squared euclidean distances.",
    # )  # type: ignore


class TSNE(SklearnWrapper, TSNEOperation):
    """Scikit-learn's TSNE wrapper for DashAI."""

    SCHEMA = TSNESchema
    DESCRIPTION = "T-distributed Stochastic Neighbor Embedding."
