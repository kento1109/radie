import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class EntityPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_state):
        pooled_output = self.dense(hidden_state)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is customized BertForSequenceClassification
    """
    def __init__(self, config, loss_fct=None, output_type='[CLS]'):
        # n_entity_markers : the number of span embedded.
        # e.g. 4 entity markers are embedded in relation extraction task
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.output_type = output_type
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler(config) if output_type.startswith(
            'entity') else None
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        # self.last_weight = torch.nn.Linear(config.hidden_size, 1, bias=False)
        # self.last_bias = nn.Parameter(torch.zeros(config.num_labels).float())
        self.loss_fct = loss_fct

    def forward(self,
                input_ids,
                token_type_ids=None,
                entity_mask=None,
                attention_mask=None,
                labels=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = None
        if self.output_type == '[CLS]':
            # Last layer hidden-state of the first token of the sequence
            pooled_output = outputs[1]

        elif self.output_type.startswith('entity'):
            # sanity check
            if entity_mask is None:
                raise ValueError('entity_mask must be given')
            # Sequence of hidden-states at the output of the last layer
            sequence_output = outputs[0]
            _batch_size = sequence_output.size(0)

            # sanity check
            # _device = entity_mask.device
            # assert all(
            #     torch.count_nonzero(entity_mask, dim=1) ==
            #     torch.zeros(_batch_size, device=_device) +
            #     self.n_target_entity_markers)

            # repeat elements of a tensor
            # https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html
            repeated_entity_mask = entity_mask.repeat_interleave(
                self.config.hidden_size).view(_batch_size, -1,
                                              self.config.hidden_size)

            n_entitity_mask = entity_mask.sum(1).repeat_interleave(
                self.config.hidden_size).view(_batch_size,
                                              self.config.hidden_size)

            # select entity hidden state from sequence output
            # then, average pooling is operated
            entity_output = (sequence_output *
                             repeated_entity_mask).sum(1) / n_entitity_mask
            pooled_output = self.entity_pooler(entity_output)

        if pooled_output is None:
            raise ValueError('pooled output is None')

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # logits = self.last_weight(pooled_output) + self.last_bias

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
