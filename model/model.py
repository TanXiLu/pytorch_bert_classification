# coding=utf-8
from transformers import BertModel
import torch.nn as nn


class BertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__()
        self.config = config
        self.bert_dim = config.bert_dim
        self.bert_encoder = BertModel.from_pretrained(config.bert_dir)

        self.dropout = nn.Dropout(config.dropout_prob)
        self.linear = nn.Linear(self.bert_dim, config.num_tags)

    def forward(self, data):
        bert_outputs = self.bert_encoder(
            input_ids=data['token_ids'],
            attention_mask=data['attention_masks'],
            token_type_ids=data['token_type_ids']
        )
        seq_out = bert_outputs[1]
        seq_out = self.dropout(seq_out)
        seq_out = self.linear(seq_out)
        return seq_out
