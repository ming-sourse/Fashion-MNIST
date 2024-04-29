import numpy as np
import matplotlib.pyplot as plt
import pickle
from NN import NeuralNetwork


# def load_model(file_name):
#     with open(file_name, 'rb') as f:
#         model = pickle.load(f)
#     return model

# 绘制权重和偏置的直方图
def plot_distribution(params, layer_num):
    for param_type, values in params.items():
        plt.figure(figsize=(10, 4))
        plt.hist(values.flatten(), bins=50, alpha=0.75)
        plt.title(f"Distribution of {param_type} in Layer {layer_num}")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

# 绘制热力图
def plot_heatmap(weights, title="Heatmap of Weights"):
    plt.figure(figsize=(8, 6))
    plt.imshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Neurons")
    plt.ylabel("Input features")
    plt.show()

# 可视化第一层权重
def visualize_first_layer_weights(weights):
    num_neurons = weights.shape[1]
    side = int(np.sqrt(weights.shape[0]))
    if side * side != weights.shape[0]:
        print("First layer weights cannot be reshaped into a square.")
        return
    fig, axes = plt.subplots(1, num_neurons, figsize=(num_neurons * 1.5, 2))
    for i, ax in enumerate(axes):
        weight_img = weights[:, i].reshape(side, side)
        ax.imshow(weight_img, cmap='gray')
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    model_path = 'best_model.pkl'
    model = NeuralNetwork(input_size=784, hidden_sizes=[128, 64], output_size=10, activations=['relu', 'relu', 'softmax'])
    model.load_model(model_path)

    # 遍历模型中的所有权重和偏置
    for i in range(1, 4):  # 假设有三层
        W = model.params[f'W{i}']
        b = model.params[f'b{i}']
        
        # 绘制权重和偏置的直方图
        # plot_distribution({'Weights': W, 'Biases': b}, i)
        
        # 绘制权重的热力图
        plot_heatmap(W, f"Heatmap of Weights for Layer {i}")
        
        # 如果是第一层且输入是图像，可视化权重
        # if i == 1:
        #     visualize_first_layer_weights(W)
