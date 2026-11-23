# PyTorch 手写数字识别与对照实验

使用 scikit-learn Digits 数据集练习数据处理、神经网络训练、验证集选模和错例分析。CPU 即可运行，无需单独申请数据集。

## 快速运行（Windows / PowerShell）

下载仓库 ZIP 并解压，在包含 `main.py` 的目录打开终端。已验证环境：Windows、Python 3.11、PyTorch 2.15.0+cpu、scikit-learn 1.9.1。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

安装依赖需要网络。数据随 scikit-learn 提供，训练时无需网络。在 PyCharm 中，将解释器选为项目 `.venv\Scripts\python.exe`，直接运行 `main.py` 即可使用默认配置（64 个隐藏单元、50 轮）。

macOS / Linux 的虚拟环境解释器路径是 `.venv/bin/python`，CPU 依赖清单针对本次 Windows 环境；其他平台需安装适用于其平台的 PyTorch。

## 四组对照实验

在项目目录依次执行，模型每次从头初始化：

```powershell
.\.venv\Scripts\python.exe main.py --hidden-dim 16 --epochs 15
.\.venv\Scripts\python.exe main.py --hidden-dim 64 --epochs 15
.\.venv\Scripts\python.exe main.py --hidden-dim 16 --epochs 50
.\.venv\Scripts\python.exe main.py --hidden-dim 64 --epochs 50
```

也可以修改 `main.py` 顶部的 `HIDDEN_DIM` 和 `EPOCHS` 后，在 PyCharm 中点击运行。命令行参数会覆盖顶部默认值。

固定设置：随机种子 42；批大小 64；Adam；学习率 0.001；交叉熵损失；CPU 训练。

2026-11-23 在本机重新执行得到的结果：

- 隐藏单元 16、15 轮：最佳第 15 轮，验证准确率 **87.0%**，测试准确率 **91.1%**。
- 隐藏单元 64、15 轮：最佳第 15 轮，验证准确率 **93.0%**，测试准确率 **93.7%**。
- 隐藏单元 16、50 轮：最佳第 49 轮，验证准确率 **94.1%**，测试准确率 **95.2%**。
- 隐藏单元 64、50 轮：最佳第 37 轮，验证准确率 **95.9%**，测试准确率 **97.0%**（270 张中分对 262 张）。

每次运行会保存至 `results/`：

- `hidden64_epochs50_seed42.json`：配置、依赖版本、每轮训练损失和验证准确率、最佳轮次、测试准确率及全部错分样本的索引与标签。
- 同名 `.pth`：最佳模型参数和隐藏层宽度，运行时生成，不提交至 Git。

同一配置再次运行会覆盖对应结果文件。换一组 `--seed` 会生成独立文件。

## 数据与方法

Digits 共 1,797 张 8×8 灰度图片，标签为数字 0–9。像素值除以 16 归一化至 0–1，再展开为 64 维向量。使用两次分层随机划分，得到训练集 1,257 张、验证集 270 张、测试集 270 张，划分随机种子为 42。

模型为 `Linear(64, hidden_dim) → ReLU → Linear(hidden_dim, 10)`。训练仅更新训练集上的参数。每轮在验证集评价，保存验证准确率最高的模型；同分时保留最早轮次。训练结束后恢复该模型，才在测试集评价。

字符错例图将灰度值按阈值二值化，仅用于查看；模型输入保留归一化灰度值。

## 观察与局限

本次实验中，增加训练轮数提高了两种网络的最佳验证准确率；在相同的 50 轮预算下，64 个隐藏单元比 16 个隐藏单元高约 1.85 个百分点。训练损失持续下降，并不保证验证准确率持续提高。

目前只运行一个随机种子，没有统计显著性结论，不能据此认为大模型总是更好。不同依赖版本或硬件可能造成数值差异。此项目是跟随指导完成的学习实验，不是新算法或论文复现。

四组配置都显示了测试结果，便于记录教学过程；配置比较以验证集为依据。后续继续调参时应停止查看测试结果，最终采用独立保留的数据评价。现有 97.0% 是本次测试划分上的结果，不代表外部手写图片上的表现。

## 文件

```text
main.py              训练与评价入口
requirements.txt     本机依赖版本
requirements-lock.txt 完整依赖版本清单
results/*.json       四组实测训练与错分记录
.gitignore           排除环境、编辑器设置、备份和模型文件
```

数据来源：[scikit-learn Digits 文档](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)。
