# coding=utf-8
import os
import torch
import time
from utils import set_logger, multi_decode


class Trainer(object):
    def __init__(self, model, optimizer, criterion, metric, device, config,
                 train_loader, dev_loader, test_loader=None, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.device = device

        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.lr_scheduler = lr_scheduler
        #
        self.global_step = 1
        self.epochs = self.config["epochs"]
        self.log_interval = self.config["log_interval"]
        self.valid_step = self.config["valid_step"]
        self.total_step = len(self.train_loader) * self.epochs
        #
        self.save_model_name = self.config["save_model_name"]
        self.save_model_dir = self.config["save_model_dir"]
        #
        self.best_valid_micro_f1 = 0.0
        #
        if not os.path.exists(config['log_dir']):
            os.makedirs(config['log_dir'])
        self.logger = set_logger(os.path.join(config['log_dir'], config['log_save_name']))
        #
        resume = self.config["resume"]
        if resume is not None:
            self._resume_checkpoint(resume_path=resume)

    def _train_loop(self, epoch):
        self.model.train()
        running_loss = 0.

        for batch, data in enumerate(self.train_loader):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(self.device)

            outputs = self.model(data)
            loss = self.criterion(outputs, data)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if self.global_step % self.log_interval == 0:
                cur_loss = running_loss / self.log_interval
                self.logger.info(
                    "[Train] epoch：{} step:{}/{} loss：{:.6f}".format(epoch, self.global_step, self.total_step, cur_loss)
                )
                running_loss = 0.0

            # 验证
            if self.global_step % self.valid_step == 0:
                dev_loss, accuracy, micro_f1, macro_f1 = self._valid_loop(epoch)
                if macro_f1 > self.best_valid_micro_f1:
                    checkpoint = {
                        "epoch": epoch,
                        "loss": dev_loss,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict()
                    }
                    self.best_valid_micro_f1 = macro_f1
                    self._save_checkpoint(checkpoint)

            self.global_step += 1

            if batch == 5000:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _valid_loop(self, epoch):
        self.model.eval()
        total_loss = 0.0
        counts = 0
        results = []
        with torch.no_grad():
            for batch, data in enumerate(self.dev_loader):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, data)
                total_loss += loss.item()
                counts += 1
                results.append((outputs, data["labels"]))
        # 解码
        pre_values, targets = multi_decode(results)
        # 评价
        accuracy, micro_f1, macro_f1 = self.metric(pre_values, targets)
        self.logger.info(
            "[Valid] epoch：{} loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(
                epoch, total_loss/counts, accuracy, micro_f1, macro_f1))
        return total_loss/counts, accuracy, micro_f1, macro_f1

    def _save_checkpoint(self, state):
        """
        Saving checkpoints
        """
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        checkpoint_path = os.path.join(self.save_model_dir, self.save_model_name)
        torch.save(state, checkpoint_path)
        self.logger.info(f"Saving current best model: {self.save_model_name} ...")

    def _resume_checkpoint(self, resume_path):
        """
        resume from saved checkpoints
        """
        checkpoint = torch.load(resume_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(checkpoint['epoch']))

    def train(self):
        for epoch in range(self.epochs):
            self._train_loop(epoch)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        counts = 0
        results = []
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, data)
                total_loss += loss.item()
                counts += 1
                results.append((outputs, data["labels"]))
        # 解码
        pre_values, targets = multi_decode(results)
        # 评价
        accuracy, micro_f1, macro_f1 = self.metric(pre_values, targets)
        self.logger.info(
            "[Test] loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(
                total_loss/counts, accuracy, micro_f1, macro_f1))
        return pre_values, targets
