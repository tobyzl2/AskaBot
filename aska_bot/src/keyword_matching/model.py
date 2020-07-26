import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel


class BertKeywordModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        bert_model_name = model_name if model_name else "bert-base-cased"

        # layers
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.kw_outputs = nn.Linear(self.bert.config.hidden_size, self.bert.config.num_labels)

        # configs
        self.num_labels = self.bert.config.num_labels

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.kw_outputs(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1  # (batch x seq_len)
                active_logits = logits.view(-1, self.num_labels)  # (batch x seq_len, num_labels)
                ignore_index = torch.tensor(loss_fct.ignore_index).type_as(labels)

                # if active_loss, take labels, otherwise, fill with ignore index
                active_labels = torch.where(active_loss, labels.view(-1), ignore_index)  # (batch x seq_len)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss

        return logits
