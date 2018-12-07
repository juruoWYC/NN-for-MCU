'''
1.CNN参数输出脚本，自动转换为数组形式的.h代码
2.支持卷积层和全连接层的组合网络结构
3.卷积层的权重经过重新组织以便于计算：
  [x][y][c][w] -> [w][c][x][y]
  x - 卷积核长度
  y - 卷积核宽度
  c - 输入矩阵通道数
  w - 卷积核权重数
'''
import keras
    
def Weights2H(weights, file):
    CNN_weights_D = 4
    ANN_weights_D = 2
    bias_D = 1
    
    for layer in range(len(weights)):
        if layer % 2 == 0:
            file.write('//layer ' + str(layer//2+1) + '\n')
            file.write('//w' + str(layer//2+1) + '\n')
        else:
            file.write('//b' + str(layer//2+1) + '\n')
        
        if len(weights[layer].shape) == CNN_weights_D:
            w = weights[layer].shape[3]
            c = weights[layer].shape[2]
            x = weights[layer].shape[0]
            y = weights[layer].shape[1]
            
            file.write('{\n')
            for i in range(w):
                file.write('{\n')
                for j in range(c):
                    file.write('{')
                    for k in range(x):
                        file.write('{')
                        for m in range(y):
                            file.write(str(weights[layer][k][m][j][i]))
                            if m != y-1:
                                file.write(',')
                        file.write('}')
                        if k != x-1:
                            file.write(',')
                    file.write('}')
                    if j != c-1:
                        file.write(',')
                    file.write('\n')
                file.write('}')
                if i != w-1:
                    file.write(',')
                file.write('\n')
            file.write('},\n')
        elif len(weights[layer].shape) == ANN_weights_D:
            x = weights[layer].shape[0]
            y = weights[layer].shape[1]
            
            file.write('{\n')
            for i in range(y):
                file.write('{')
                for j in range(x):
                    file.write(str(weights[layer][j][i]))
                    if j != x-1:
                        file.write(',')
                file.write('}')
                if i != y-1:
                    file.write(',')
                file.write('\n')
            file.write('},\n')
        elif len(weights[layer].shape) == bias_D:
            n = weights[layer].shape[0]
            
            file.write('{\n')
            for i in range(n):
                file.write(str(weights[layer][i]))
                if i != n-1:
                    file.write(',')
            file.write('\n},\n')
            
            
if __name__ == '__main__':
    file = open('weights-CNN.h', 'w')
    model = keras.models.load_model('Model_CNN.H5')
    weights = model.get_weights()
    Weights2H(weights, file)
    file.close()