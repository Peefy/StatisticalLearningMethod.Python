
import numpy as _np
import math as _math

class Perception:
    """
    感知机
    """
    def __init__(self, yita = 0.1, w0 = 0, b0 = 0):
        """
        感知机
        """
        self.yita = yita
        self.w0 = w0
        self.b0 = b0
    
    def perception_func(self, w, x, b):
        return _np.sign(w * x + b)

    def loss_func(self, w, x, b, y):
        return y * (w * x + b)

    def train(self, xdata, ydata):
        assert len(xdata) == len(ydata)    
        for i in range(len(xdata)):
            x = xdata[i]
            y = ydata[i]
            result = self.loss_func(self.w0, x, self.b0, y)
            if result <= 0:
                self.w0, self.b0 = self.w0 + y * x, self.b0 + y
    
    def run(self, x):
        return [self.perception_func(self.w0, xx, self.b0) for xx in x] 

class SKLearnPerception:
    pass

def main():
    p = Perception()
    x = _np.array([0, 1, 2, 3])
    w = _np.array([0, 1, 2, 3])
    b = _np.array([0, -3, 2, 3])
    y = p.perception_func(w, x, b)
    print(y)
    loss = p.loss_func(w, x, b, y)
    print(loss)
    pp = Perception(w0=0, b0=0)
    xdata = [3, 4, 1]
    ydata = [1, 1, -1]
    x_test = [-1, 0, 1, 2, 3, 4, 5]
    p.train(xdata, ydata)
    y_run = p.run(x_test)
    print(y_run)

if __name__ == '__main__':
    main()
