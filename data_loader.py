import numpy as np
import struct

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))    # H是无符号短整型（两个字节）B是一个字节；H对应zero，B对应data_type，B对应dims
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))   # read()会随着文件读取自动移动，unpack返回的是元组，故要[0];I是无符号长整型（四个字节）
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)   # frombuffer从缓存创建数组，reshape为读取的形状

def load_mnist_images(filename):
    return read_idx(filename)

def load_mnist_labels(filename):
    return read_idx(filename)