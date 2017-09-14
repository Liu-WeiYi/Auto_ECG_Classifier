# Auto_ECG_Classifier
Use resnet32 structure, identify wavelet in ECG. (采用resnet32架构，识别ECG(心电图)中的波形)

Usage:
## 训练网络
python3 main.py --train_flag True

## 验证结果
python3 main.py --test_flag True

## Tips
### 采用1D的卷积层
### 1. filter的输出channels的个数自己给定
### 2. filter的输出channels确定了下一层conv的channel的输入
1. training Sample:  [batch, input_width, input_channels]
2. training Filter:  [filter_width, input_channels, output_channels]
