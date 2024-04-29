# 用字典存储网络配置和超参数
network_config = {
    "input_size": 784,  # 图像尺寸 (28x28)
    "output_size": 10,  # 类别数 (对于MNIST是10)
    "activations": ['relu', 'relu', 'softmax'],  # 激活函数
}

# 定义不同的超参数组合以用于超参数调优
hyperparameters = {
    "learning_rates": [0.001, 0.01, 0.1],
    "hidden_sizes_options": [
        [128, 64],  # 第一种隐藏层结构
        [256, 128],  # 第二种隐藏层结构
        [256, 256]  # 第三种隐藏层结构
    ],
    "reg_lambdas": [0.001, 0.01, 0.05],  # 正则化强度
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate_decay": 0.95
}


data_paths = {
    "images_path": 'Data/train-images-idx3-ubyte',
    "labels_path": 'Data/train-labels-idx1-ubyte'
}
