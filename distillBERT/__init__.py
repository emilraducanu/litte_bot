from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_distilbert": [
        "DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DistilBertConfig",
        "DistilBertOnnxConfig",
    ],
    "tokenization_distilbert": ["DistilBertTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_distilbert_fast"] = ["DistilBertTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_distilbert"] = [
        "DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DistilBertForMaskedLM",
        "DistilBertForMultipleChoice",
        "DistilBertForQuestionAnswering",
        "DistilBertForSequenceClassification",
        "DistilBertForTokenClassification",
        "DistilBertModel",
        "DistilBertPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_distilbert"] = [
        "TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDistilBertForMaskedLM",
        "TFDistilBertForMultipleChoice",
        "TFDistilBertForQuestionAnswering",
        "TFDistilBertForSequenceClassification",
        "TFDistilBertForTokenClassification",
        "TFDistilBertMainLayer",
        "TFDistilBertModel",
        "TFDistilBertPreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_distilbert"] = [
        "FlaxDistilBertForMaskedLM",
        "FlaxDistilBertForMultipleChoice",
        "FlaxDistilBertForQuestionAnswering",
        "FlaxDistilBertForSequenceClassification",
        "FlaxDistilBertForTokenClassification",
        "FlaxDistilBertModel",
        "FlaxDistilBertPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_distilbert import (
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DistilBertConfig,
        DistilBertOnnxConfig,
    )
    from .tokenization_distilbert import DistilBertTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_distilbert_fast import DistilBertTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_distilbert import (
            DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DistilBertForMaskedLM,
            DistilBertForMultipleChoice,
            DistilBertForQuestionAnswering,
            DistilBertForSequenceClassification,
            DistilBertForTokenClassification,
            DistilBertModel,
            DistilBertPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_distilbert import (
            TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDistilBertForMaskedLM,
            TFDistilBertForMultipleChoice,
            TFDistilBertForQuestionAnswering,
            TFDistilBertForSequenceClassification,
            TFDistilBertForTokenClassification,
            TFDistilBertMainLayer,
            TFDistilBertModel,
            TFDistilBertPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_distilbert import (
            FlaxDistilBertForMaskedLM,
            FlaxDistilBertForMultipleChoice,
            FlaxDistilBertForQuestionAnswering,
            FlaxDistilBertForSequenceClassification,
            FlaxDistilBertForTokenClassification,
            FlaxDistilBertModel,
            FlaxDistilBertPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
