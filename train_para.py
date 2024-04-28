# train.py
import numpy as np
from NN import NeuralNetwork  # 确保你的神经网络类文件名和类名正确
from data_loader import load_mnist_images, load_mnist_labels  # 加载数据的函数
from config import network_config, hyperparameters, data_paths  # 从config.py导入配置

def load_and_prepare_data():
    # 加载数据
    images = load_mnist_images(data_paths['images_path'])
    labels = load_mnist_labels(data_paths['labels_path'])
    # 转换为向量并归一化
    images = images.reshape(images.shape[0], -1) / 255.0
    # 划分训练集和验证集
    X_train, X_val = images[:50000], images[50000:]
    y_train, y_val = labels[:50000], labels[50000:]
    return X_train, y_train, X_val, y_val

def train_and_evaluate(X_train, y_train, X_val, y_val, params):
    # 初始化神经网络
    network = NeuralNetwork(
        input_size=network_config['input_size'],
        hidden_sizes=params['hidden_sizes'],
        output_size=network_config['output_size'],
        activations=network_config['activations']
    )
    
    # 训练网络
    network.train(
        X_train, y_train, X_val, y_val,
        num_epochs=params['num_epochs'],
        learning_rate=params['learning_rate'],
        reg_lambda=params['reg_lambda'],
        batch_size=hyperparameters['batch_size'],
        learning_rate_decay=hyperparameters['learning_rate_decay']
    )
    
    # 评估模型
    val_accuracy = network.evaluate_accuracy(X_val, y_val)
    return network, val_accuracy

def main():
    X_train, y_train, X_val, y_val = load_and_prepare_data()
    
    best_accuracy = 0
    best_network = None
    best_params = None
    
    # 超参数搜索
    for lr in hyperparameters['learning_rates']:
        for hidden_sizes in hyperparameters['hidden_sizes_options']:
            for reg_lambda in hyperparameters['reg_lambdas']:
                params = {
                    'learning_rate': lr,
                    'hidden_sizes': hidden_sizes,
                    'reg_lambda': reg_lambda,
                    'num_epochs': hyperparameters['num_epochs']
                }
                print(f"Testing with params: {params}")
                network, accuracy = train_and_evaluate(X_train, y_train, X_val, y_val, params)
                print(f"Validation Accuracy: {accuracy * 100:.2f}%")
                
                # 更新最佳参数
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = params
                    best_network = network
                    best_network.save_model('best_model.pkl')
                    print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")
    
    print(f"Best validation accuracy: {best_accuracy * 100:.2f}% with parameters {best_params}")

if __name__ == '__main__':
    main()
