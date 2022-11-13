import torch
from transformers import (
    BertForSequenceClassification,
    BertPreTrainedModel,
    BertLayer,
    BertLMHeadModel,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForMultipleChoice,
    BertForTokenClassification,
    BertForQuestionAnswering,
)

class LSBertPreTrainedModel(BertPreTrainedModel):
    @classmethod
    def from_pretrained(self, *args, training_args, model_args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        return model


class LSBertForSequenceClassification(
    LSBertPreTrainedModel, BertForSequenceClassification
):
    """from BertForSequenceClassification"""


class LSBertLMHeadModel(LSBertPreTrainedModel, BertLMHeadModel):
    """from BertLMHeadModel"""


class LSBertForMaskedLM(LSBertPreTrainedModel, BertForMaskedLM):
    """from BertForMaskedLM"""


class LSBertForNextSentencePrediction(
    LSBertPreTrainedModel, BertForNextSentencePrediction
):
    """from BertForNextSentencePrediction"""


class LSBertForMultipleChoice(LSBertPreTrainedModel, BertForMultipleChoice):
    """from BertForMultipleChoice"""


class LSBertForTokenClassification(
    LSBertPreTrainedModel, BertForTokenClassification):
    """from BertForTokenClassification"""


class LSBertForQuestionAnswering(LSBertPreTrainedModel, BertForQuestionAnswering):
    """from BertForQuestionAnswering"""
