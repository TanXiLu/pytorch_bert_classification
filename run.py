# coding=utf-8
import torch
from config import ArgConfig
from transformers import BertTokenizer
from utils import prepare_device, set_seed, read_json
from data_loader import get_loader
from trainer import Trainer
from model import BertForSequenceClassification, cal_loss, get_classification_report, get_metrics


def run(config):
    set_seed(seed=42)
    tokenizer = BertTokenizer.from_pretrained(config['bert_dir'])
    train_loader = get_loader(config, prefix='train', tokenizer=tokenizer)
    dev_loader = get_loader(config, prefix='val', tokenizer=tokenizer)
    test_loader = get_loader(config, prefix='test', tokenizer=tokenizer)
    # 模型
    device, device_ids = prepare_device(config['n_gpu'])
    model = BertForSequenceClassification(config)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # 优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])
    # 评价函数
    metric = get_metrics
    # 损失函数
    criterion = cal_loss

    trainer = Trainer(model, optimizer, criterion, metric, device, config, train_loader, dev_loader, test_loader)
    # 训练
    if config["do_train"]:
        trainer.train()
    # 测试
    if config["do_test"]:
        if not config["do_train"]:
            assert config.resume is not None, 'make sure resume is not None'
        tag2id, _ = read_json(config.data_path, config.label_name)
        labels = [tag for tag, _ in sorted(tag2id.items(), key=lambda x: x[1])]
        outputs, targets = trainer.evaluate()
        print(get_classification_report(outputs, targets, labels))


if __name__ == '__main__':
    conf = ArgConfig()
    conf.batch_size = 24
    conf.do_train = False
    conf.do_test = True
    run(conf)
