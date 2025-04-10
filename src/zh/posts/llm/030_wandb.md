---
icon: lightbulb
sidebar: false
date: 2025-03-10
prev: ./031_qlora
next: ./029_unsloth_grpo
category:
  - LLM
tag:
  - wandb
---
# 深度学习轻量级可视化工具wandb
- wandb 介绍
- 常用函数
- How to Use
<!-- more -->
## 1. what
wandb（Weights & Biases）是一个与Tensorboard类似的模型训练可视化平台。不过，相比较TensorBoard而言，Wandb更加的强大，主要体现在以下的几个方面：
- 4项核心功能（wandb并不单纯的是一款数据可视化工具。它具有更为强大的模型和数据版本管理。此外，还可以对你训练的模型进行调优）
    - Dashboard 看板：跟踪训练过程，给出可视化结果
    - Reports 报告：保存和共享训练过程中一些细节、有价值的信息
    - Sweeps 超参调优：使用超参数调优来优化你训练的模型
    - Artifacts 工具：数据集和模型版本控制，可以自己搭建pipeline，实现保存数据集和模型以及评估结果的流程
- 强大兼容性（兼容各种框架和环境）：能够和Jupyter、TensorFlow、Pytorch、Keras、Scikit、fast.ai、LightGBM、XGBoost一起结合使用，可以与各种机器学习基础架构配合使用：AWS，GCP，Kubernetes，Azure和本地机器。
- 复现模型：Wandb更有利于复现模型。这是因为Wandb不仅记录指标，还会记录超参数和代码版本。
- 自动上传云端：Wandb可以让你便捷地查看你制作的所有模型，你就不必花费大量时间来重新运行旧实验。
- 快速、灵活的集成：只需5分钟即可把Wandb加到自己的项目。下载Wandb免费的开源Python包，然后在代码中插入几行，以后你每次运行模型都会得到记录完备的指标和记录。
- 集中式指示板：Wandb提供同样的集中式指示板。不管在哪里训练模型，不管是在本地机器、实验室集群还是在云端实例；
这样就不必花时间从别的机器上复制TensorBoard文件。
- 强大的表格：对不同模型的结果进行搜索、筛选、分类和分组。可以轻而易举地查看成千上万个模型版本，并找到不同任务的最佳模型。
- 运行的同时分析系统硬件的情况如：CPU和GPU使用率

![](../../../assets/030_wandb.png)

## 2. 常用函数
- wandb.init() # 初始化
    - project: 项目名称
- wandb.config # 超参数配置
- wandb.log() # 记录训练循环中持续变化的指标
- wandb.Image() # 图像显示
- wandb.save() # 保存模型
- wandb.watch()
## 3. How to Use
### 3.1. 安装wangdb库
```python
!pip install wandb
```

### 3.2. 官网注册获取API Key
官网：[https://wandb.ai/](https://wandb.ai/)

一般用google账户或者github账户注册都行， 注册时注意保存username
注册成功会在这个页面有注册信息展示，api key也是在这个页面获取的

### 3.3. 在终端运行wandb login
```python
wandb.login()
```
输入API key

自动登录设置：运行 wandb login命令后， 会在/root/.netrc 下生成.netrc文件， 修改这个文件，这里的password就是第二步获取的api key，之后就不用输密码了

### 3.4. 初始化
```python
import wandb

wandb.init(project="my-project")
```
### 3.5. 声明超参数
```python
wandb.config.dropout = 0.2
wandb.config.hidden_layer_size = 128
```
### 3.6. 记录日志
```python
def my_train_loop():
    for epoch in range(10):
        loss = 0 # change as appropriate :)
        wandb.log({'epoch': epoch, 'loss': loss})
```

### 3.7. 发出报警
```
acc_threshold = 0.3
if accuracy <= acc_threshold:
        wandb.alert(
            title='Low Accuracy',
            text=f'Accuracy {accuracy} at step {training_step} is below the acceptable theshold, {acc_threshold}',
        )
```

### 3.8. 保存文件
```python
wandb.save("mymodel.h5")
```

### 3.9. 查看wandb看板
打开运行结果里有火箭那一行的链接，浏览器中出现界面就说明成功了。