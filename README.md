# 实现FGSM对抗攻击方法
## paper pdf 
```bash
cd paper
```

# Installation
## Install python libraries.
```bash
pip install -r requirements.txt
```

# Dataset
## Mstar
```bash
cd data
```

# Usage
## 训练resnet50模型
```bash
cd src
python train.py
```
## Attack on Mstar
```bash
python fgsm.py
```
# Other
+ 模型训练权重(需要先训练)
```bash
cd checkpoint
```
+ 对抗样本(先攻击，后生成)
```bash
cd data/Mstar/fgsm
```
# <table><tr><td bgcolor="yellow"><font color=Slategrey>提交生成的对抗样本文件夹</font></tr></td></table>
