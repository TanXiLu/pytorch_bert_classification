# coding=utf-8
import os


class ArgConfig(object):
    def __init__(self):
        self.n_gpu = 1
        self.batch_size = 8
        self.epochs = 10
        self.log_interval = 20
        self.valid_step = 200

        # 路径和名称
        self.dataset = 'cnews'
        self.label_name = 'labels'
        self.resume = os.path.join('./checkpoints/', self.dataset, 'best.pt')
        # self.resume = None

        self.save_model_dir = './checkpoints/' + self.dataset
        self.save_model_name = 'best.pt'

        self.log_dir = './logs/' + self.dataset
        self.log_save_name = 'log.log'

        self.data_path = './data/' + self.dataset
        self.result_dir = './result/' + self.dataset
        self.bert_dir = './model_hub/chinese-bert-wwm-ext/'

        # 模型超参数
        self.dropout_prob = 0.1
        self.bert_dim = 768
        self.num_tags = 10
        self.max_seq_len = 512

        # 优化器超参数
        self.learning_rate = 3e-5

        # 训练/测试
        self.do_train = True
        self.do_test = False

    def __getitem__(self, item):
        return getattr(self, item)
