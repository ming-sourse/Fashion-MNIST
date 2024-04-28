import numpy as np
import pickle
import json

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activations):
        # 初始化网络参数
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activations = activations
        self.params = self.initialize_params()
        self.history = {
            'train_loss': [],
            'val_loss': [],  # 验证集损失
            'val_accuracy': []
        }
        
    def initialize_params(self):
        # 初始化权重和偏置参数
        np.random.seed(42)  # 设置随机种子以确保结果可重复
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]  # 网络各层大小的列表
        params = {}
        for i in range(len(layer_sizes) - 1):
            params['W' + str(i+1)] = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01  # 权重初始化为小随机数
            params['b' + str(i+1)] = np.zeros((1, layer_sizes[i+1]))  # 偏置初始化为0
        return params

    def relu(self, Z):
        # ReLU激活函数
        return np.maximum(0, Z)

    def softmax(self, Z):
        # Softmax激活函数
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # -max是为了防止数值溢出；keepdim保证max结果为(m, 1),而不是(m,);Z为(m, n),m为样本量，n为特征量
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_pred, y_true):
        # 交叉熵损失函数
        m = y_true.shape[0]  # y_true应该是(m,)
        log_likelihood = -np.log(y_pred[range(m), y_true])  # numpy的高级索引功能，y_pred是m*10
        loss = np.sum(log_likelihood) / m
        return loss

    def forward(self, x):
        # 前向传播过程
        cache = {'A0': x}  # 保存前向传播的中间结果，Z和A均要保存，用于反向传播，x是矩阵
        for i in range(len(self.hidden_sizes) + 1):
            W = self.params['W' + str(i+1)]
            b = self.params['b' + str(i+1)]
            Z = cache['A' + str(i)].dot(W) + b  # 线性变换
            if i < len(self.hidden_sizes):
                if self.activations[i] == 'relu':
                    A = self.relu(Z)  # ReLU激活函数
            else:
                A = self.softmax(Z)  # 输出层使用Softmax
            cache['Z' + str(i+1)] = Z
            cache['A' + str(i+1)] = A
        return A, cache

    def backward(self, y, cache):
        # 反向传播过程
        grads = {}
        m = y.shape[0]
        y_pred = cache['A' + str(len(self.hidden_sizes) + 1)]
        y_pred[range(m), y] -= 1   # 这一步是计算最后一层的dz，其余的dz都为relu求导
        y_pred /= m
        for i in reversed(range(len(self.hidden_sizes) + 1)):
            dZ = y_pred
            dW = cache['A' + str(i)].T.dot(dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            if i > 0:
                dA_prev = dZ.dot(self.params['W' + str(i+1)].T)
                if self.activations[i-1] == 'relu':
                    dA_prev[cache['Z' + str(i)] <= 0] = 0
            y_pred = dA_prev
            grads['W' + str(i+1)] = dW
            grads['b' + str(i+1)] = db
        return grads

    def update_params(self, grads, learning_rate):
        # 更新网络参数
        for i in range(len(self.hidden_sizes) + 1):
            self.params['W' + str(i+1)] -= learning_rate * grads['W' + str(i+1)]  # 根据梯度和学习率更新权重
            self.params['b' + str(i+1)] -= learning_rate * grads['b' + str(i+1)]  # 根据梯度和学习率更新偏置

    def train(self, X_train, y_train, X_val, y_val, num_epochs, learning_rate, reg_lambda, batch_size, learning_rate_decay):
        best_val_acc = 0
        best_params = {}
        n_samples = X_train.shape[0]

        for epoch in range(num_epochs):
            # Shuffle the training data
            indices = np.arange(n_samples)  
            np.random.shuffle(indices)    # [0~n_samples]随机打乱，防止总是以相同的数据顺序进行训练
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # Forward and backward passes
                y_pred, cache = self.forward(X_batch)
                loss = self.cross_entropy_loss(y_pred, y_batch)
                # Add regularization to the loss
                for i in range(len(self.hidden_sizes) + 1):
                    W = self.params['W' + str(i+1)]
                    loss += 0.5 * reg_lambda * np.sum(W**2)

                grads = self.backward(y_batch, cache)
                # Add regularization to the gradients
                for i in range(len(self.hidden_sizes) + 1):
                    grads['W' + str(i+1)] += reg_lambda * self.params['W' + str(i+1)]

                self.update_params(grads, learning_rate)

            # Learning rate decay
            learning_rate *= learning_rate_decay

            # Validation
            val_acc = self.evaluate_accuracy(X_val, y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = self.params.copy()
                # self.save_model('best_model.pkl')

            self.history['train_loss'].append(loss)  # 存储训练损失
            # 计算验证集损失和准确率
            y_val_pred, _ = self.forward(X_val)
            val_loss = self.cross_entropy_loss(y_val_pred, y_val)  # 计算验证集损失
            self.history['val_loss'].append(val_loss)  # 存储验证集损失
            val_acc = self.evaluate_accuracy(X_val, y_val)  # 计算验证集准确率
            self.history['val_accuracy'].append(val_acc)  # 存储验证集准确率

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}, Validation Accuracy: {val_acc}")

        self.params = best_params

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def evaluate_accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def save_model(self, file_name):
        model_info = {
            'architecture': {
                'input_size': self.input_size,
                'hidden_sizes': self.hidden_sizes,
                'output_size': self.output_size,
                'activations': self.activations
            },
            'weights': self.params  # 假设这个方法能返回所有层的权重
        }
        with open(file_name, 'wb') as f:
            pickle.dump(model_info, f)  

    def load_model(self, file_name):
        with open(file_name, 'rb') as f:
            model_info = pickle.load(f)
        
        self.input_size=model_info['architecture']['input_size']
        self.hidden_sizes=model_info['architecture']['hidden_sizes']
        self.output_size=model_info['architecture']['output_size']
        self.activations=model_info['architecture']['activations']
    
        self.params = model_info['weights']  # 假设这个方法能设置所有层的权重
        

    # def load_model(self, file_name):
    #     with open(file_name, 'rb') as f:
    #         self.params = pickle.load(f)



# 代码中用到numpy的广播机制（NumPy Broadcasting）
# NumPy的广播规则允许在满足特定条件的情况下，对形状不同的数组进行数学运算。具体规则如下：
# 1. 如果两个数组的维数不同，将维数较小的数组形状前面补1，直至两者维数相同。
# 2. 对于任意一个维度，如果一个数组在该维度上的大小为1，而另一个数组在该维度上的大小大于1，则形状较小的数组会沿该维度复制扩展以匹配另一个数组的形状。
# 3. 如果在任何一个维度上，两个数组的大小都不是1，且不相等，则无法广播，会引发错误。