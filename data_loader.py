# coding=utf-8
import os
import torch
from utils import read_json
from torch.utils.data import Dataset, DataLoader


class CustomDataSet(Dataset):
    def __init__(self, xpath):
        self.data = self.load_data(xpath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def load_data(xpath):
        data = []
        with open(xpath, 'r', encoding='utf-8') as f:
            raw_data = f.readlines()
            for d in raw_data:
                d = d.strip().split('\t')
                text = d[1]
                label = d[0]
                data.append((text, label))
        return data


class Collate(object):
    def __init__(self, tokenizer, max_len, tag2id):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2id = tag2id

    def collate_fn(self, batch):
        batch_labels = []
        batch_token_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        for i, (text, label) in enumerate(batch):
            output = self.tokenizer.encode_plus(
                text=text,
                max_length=self.max_len,
                padding="max_length",
                truncation="longest_first",
                return_token_type_ids=True,
                return_attention_mask=True
            )

            token_ids = output["input_ids"]
            token_type_ids = output["token_type_ids"]
            attention_mask = output["attention_mask"]
            # 前面已经限制了长度
            batch_token_ids.append(token_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(self.tag2id[label])

        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long)
        attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        batch_data = {
            "token_ids": batch_token_ids,
            "attention_masks": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": batch_labels
        }
        return batch_data


def get_loader(config, prefix, tokenizer):
    max_len = config.max_seq_len
    tag2id, _ = read_json(config.data_path, config.label_name)
    collate = Collate(tokenizer, max_len, tag2id)
    filename = os.path.join(config["data_path"], '{}.{}.txt'.format(config['dataset'], prefix))
    dataset = CustomDataSet(filename)
    data_loader = DataLoader(dataset,
                             batch_size=config.batch_size,
                             shuffle=True,
                             collate_fn=collate.collate_fn)
    return data_loader

