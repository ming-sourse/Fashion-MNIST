import numpy as np
from NN import NeuralNetwork  
from data_loader import load_mnist_images, load_mnist_labels  

def load_and_prepare_data(images_path, labels_path):
    # 加载测试数据
    images = load_mnist_images(images_path)
    labels = load_mnist_labels(labels_path)

    # 转换为向量并归一化
    images = images.reshape(images.shape[0], -1) / 255.0

    return images, labels

def test_model(model_path, images_path, labels_path):
    # 创建一个神经网络实例
    network = NeuralNetwork(input_size=784, hidden_sizes=[128, 64], output_size=10, activations=['relu', 'relu', 'softmax'])

    # 加载训练好的模型
    network.load_model(model_path)

    # 加载并准备数据
    X_test, y_test = load_and_prepare_data(images_path, labels_path)

    # 评估在测试集上的准确率
    accuracy = network.evaluate_accuracy(X_test, y_test)
    print(f"{accuracy}")
    # print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    model_path = 'best_model.pkl'  # 模型文件路径
    images_path = 'Data/t10k-images-idx3-ubyte'  # 测试集图像文件路径
    labels_path = 'Data/t10k-labels-idx1-ubyte'  # 测试集标签文件路径

    test_model(model_path, images_path, labels_path)
