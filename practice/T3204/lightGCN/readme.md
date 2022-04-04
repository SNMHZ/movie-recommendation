# 22.03.30 light-GCN for movie rec
## How to run

### Train
```
python preprocess.py
python main.py

```
전체 데이터로 학습하기를 원한다면,
dataloader.py 의 425행
```
train_file = path + '/train.csv'
```
를
```
train_file = path + '/full.csv'
```
로 변경

### Inference
```
python inference.py
```

## Hyper parameters
### 03.30
1. epoch = 100
2. learning_rate = 0.001
3. LB score = 0.106

### 03.31
1. epoch = 303
2. learning_rate = 0.001
3. LB score = 0.1158

### 04.01
1. epoch = 500
2. learning_rate = 0.001
3. LB score = 0.1178

