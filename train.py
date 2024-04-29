import numpy as np
from NN import NeuralNetwork  # 导入刚才定义的 NeuralNetwork 类
from data_loader import load_mnist_images, load_mnist_labels  
import matplotlib.pyplot as plt

def plot_history(network):
    epochs = range(1, len(network.history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, network.history['train_loss'], label='Training Loss')
    plt.plot(epochs, network.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, network.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # 加载数据
    images_path = 'Data/train-images-idx3-ubyte'
    labels_path = 'Data/train-labels-idx1-ubyte'
    images = load_mnist_images(images_path)
    labels = load_mnist_labels(labels_path)
    images = images.reshape(images.shape[0], -1)  # 转换为向量
    images = images.astype(np.float32) / 255.0   # 标准化

    # 数据划分为训练集和验证集
    # 假设我们使用前50000张为训练集，后10000张为验证集
    X_train, X_val = images[:50000], images[50000:]
    y_train, y_val = labels[:50000], labels[50000:]


    X_train = images
    y_train = labels
    # 初始化神经网络
    print("Initializing the neural network...")
    network = NeuralNetwork(input_size=784, hidden_sizes=[256, 256], output_size=10, activations=['relu', 'relu', 'softmax'])

    # 设置训练参数
    num_epochs = 10
    learning_rate = 0.1
    reg_lambda = 0.001
    batch_size = 32
    learning_rate_decay = 0.95

    # 训练神经网络
    print("Training the network...")
    network.train(X_train, y_train, X_val, y_val, num_epochs=num_epochs, learning_rate=learning_rate, reg_lambda=reg_lambda, batch_size=batch_size, learning_rate_decay=learning_rate_decay)
    
    # 评估模型在验证集上的性能
    print("Evaluating the network...")
    val_accuracy = network.evaluate_accuracy(X_val, y_val)
    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

    # 保存模型
    print("Saving the model...")
    network.save_model('best_model.pkl')
    print("Model saved to 'best_model.pkl'.")

    plot_history(network)

if __name__ == '__main__':
    main()
