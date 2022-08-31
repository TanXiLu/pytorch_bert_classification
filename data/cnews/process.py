# coding=utf-8
import json
labels = []

with open('./cnews.train.txt', 'r', encoding='utf-8') as fp:
    lines = fp.read().strip().split('\n')
    for line in lines:
        line = line.split('\t')
        labels.append(line[0])


label2id = {}
id2label = {}
labels = sorted(set(labels))
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
print(label2id)

json.dump([label2id, id2label], open("./labels.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
