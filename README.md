## 中文文本多分类
本项目是基于bert的中文文本分类任务

### 相关说明
```
--checkpoints: 保存模型和加载模型
--data: 存放数据和数据初步处理
--logs: 存放日志
--model: 存放模型、损失函数、评估函数
--model_hub: 存放预训练模型bert
--utils: 解码函数和辅助函数
--config.py: 相关配置文件
--data_loader.py: 处理模型所需的输入数据格式
--inference.py: 模型推理
--run.py: 训练和测试的运行文件
--trainer.py: 训练和测试封装的类
```

在hugging face上预先下载好预训练的bert模型，存放在model_hub文件夹下

### 数据来源
使用的数据集是THUCNews，数据地址：[THUCNews](https://github.com/gaussic/text-classification-cnn-rnn)

数据集划分如下：
- 训练集： 5000*10
- 验证集 500*10
- 测试集：1000*10

**一般步骤**
1. data下新建cnews文件夹，将数据存放该文件夹下。在文件夹下新建process.py，目的是获取标签并存储在labels.json中
2. 在run.py中，可以在主函数下，调节参数 do_train/do_test 自由切换是训练+测试评估，还是单独的训练，或者还是单独的测试评估
3. 运行inference.py，可以看到文本预测的类别和概率，可在部署中辅助使用。