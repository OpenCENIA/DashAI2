from sklearn.decomposition import (
    LatentDirichletAllocation as LatentDirichletAllocationOperation,
)

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    enum_field,
    float_field,
    int_field,
    none_type,
    schema_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class LatentDirichletAllocationSchema(BaseSchema):
    n_components: schema_field(
        int_field(ge=1),
        10,
        "Number of topics to extract.",
    )  # type: ignore
    doc_topic_prior: schema_field(
        none_type(float_field()),
        None,
        "Prior of document topic distribution theta.",
    )  # type: ignore
    topic_word_prior: schema_field(
        none_type(float_field()),
        None,
        "Prior of topic word distribution beta.",
    )  # type: ignore
    learning_method: schema_field(
        enum_field(["batch", "online"]),
        "batch",
        "Method used to update topic distribution.",
    )  # type: ignore
    learning_decay: schema_field(
        float_field(gt=0.0, le=1.0),
        0.7,
        "Control the learning rate in the online learning method.",
    )  # type: ignore
    learning_offset: schema_field(
        float_field(),
        10.0,
        "A (positive) parameter that downweights early iterations in online learning.",
    )  # type: ignore
    max_iter: schema_field(
        int_field(gt=0),
        10,
        "Maximum number of iterations.",
    )  # type: ignore
    batch_size: schema_field(
        int_field(gt=0),
        128,
        "Number of documents to use in each EM iteration.",
    )  # type: ignore
    evaluate_every: schema_field(
        int_field(),
        -1,
        "How often to evaluate perplexity.",
    )  # type: ignore
    total_samples: schema_field(
        int_field(gt=0),
        1e6,
        "Total number of documents.",
    )  # type: ignore
    perp_tol: schema_field(
        float_field(gt=0.0),
        1e-1,
        "Perplexity tolerance.",
    )  # type: ignore
    mean_change_tol: schema_field(
        float_field(gt=0.0),
        1e-3,
        "Stopping tolerance for updating document topic distribution.",
    )  # type: ignore
    max_doc_update_iter: schema_field(
        int_field(gt=0),
        100,
        "Maximum number of iterations for updating document topic distribution.",
    )  # type: ignore
    n_jobs: schema_field(
        none_type(int_field()),
        None,
        "Number of jobs to run in parallel.",
    )  # type: ignore
    verbose: schema_field(
        int_field(),
        0,
        "Verbosity level.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "Pass an int for reproducible results across multiple function calls.",
    )  # type: ignore


class LatentDirichletAllocation(SklearnWrapper, LatentDirichletAllocationOperation):
    """Scikit-learn's LatentDirichletAllocation wrapper for DashAI."""

    SCHEMA = LatentDirichletAllocationSchema
    DESCRIPTION = "Latent Dirichlet Allocation with online variational Bayes algorithm."
