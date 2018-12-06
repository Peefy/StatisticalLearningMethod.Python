import numpy as _np
import math as _math
import matplotlib.pyplot as plt


class Perception:
    """
    感知机
    """

    def __init__(self, yita=0.1, w0=0, b0=0):
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
    """
    f(x) = sign(w·x+b)
    """
    def __init__(self):
        """
        f(x) = sign(w·x+b)
        """
        pass

    def demo(self):
        from sklearn.datasets import make_classification
        from sklearn.linear_model import Perceptron
        from sklearn import datasets
        # n_samples:生成样本的数量
        # n_features=2:生成样本的特征数，特征数=n_informative（） + n_redundant + n_repeated
        # n_informative：多信息特征的个数
        # n_redundant：冗余信息，informative特征的随机线性组合
        # n_clusters_per_class ：某一个类别是由几个cluster构成的
        x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
        x_data_train = x[:800, :]
        x_data_test = x[800:, :]
        y_data_train = y[:800]
        y_data_test = y[800:]

        # 正例和反例
        positive_x1 = [x[i, 0] for i in range(1000) if y[i] == 1]
        positive_x2 = [x[i, 1] for i in range(1000) if y[i] == 1]
        negetive_x1 = [x[i, 0] for i in range(1000) if y[i] == 0]
        negetive_x2 = [x[i, 1] for i in range(1000) if y[i] == 0]

        # 划分训练集、测试集：
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        # 定义感知机
        clf = Perceptron(fit_intercept=False, n_iter=30, shuffle=False)
        clf.fit(x_train, y_train)
        print(clf.score(x_test, y_test))
        from matplotlib import pyplot as plt
        # 画出正例和反例的散点图
        plt.scatter(positive_x1, positive_x2, c='red')
        plt.scatter(negetive_x1, negetive_x2, c='blue')
        # 画出超平面（在本例中即是一条直线）
        import numpy as np
        line_x = np.arange(-4, 4)
        line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
        plt.plot(line_x, line_y)
        plt.show()


class SKLearnMLPClassifier:
    """
    多层感知机
    """
    def __init__(self):
        pass

    def demo(self):
        # 载入数据集合
        from sklearn.datasets import load_digits
        digits = load_digits()
        # 数据标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(digits.data)
        x_scaled = scaler.transform(digits.data)
        # 数据分类
        x = x_scaled
        y = digits.target
        # 划分训练集、测试集：
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        # 调用sklearn，使用感知机预测：
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), activation='logistic', max_iter=100)
        mlp.fit(x_train, y_train)
        # 进行预测，并观察效果：
        from sklearn.metrics import classification_report
        predicted = mlp.predict(x_test)
        print(classification_report(y_test, predicted))
        # 进行调参，并观察参数改变对预测效果的影响：
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV
        pipeline = Pipeline([
            ('mlp', MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=100))
        ])
        parameters = {
            'mlp__activation': ('identity', 'logistic', 'tanh', 'relu'),
            'mlp__solver': ('lbfgs', 'sgd', 'adam')
        }
        grid_search = GridSearchCV(pipeline, parameters, verbose=1, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print('最佳效果：%0.3f' % grid_search.best_score_)
        print('最优参数：')
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print('\t%s: %r' % (param_name, best_parameters[param_name]))

        predictions = grid_search.predict(x_test)
        print(classification_report(y_test, predictions))

def myPerception():
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
    # plt.plot(xdata, ydata)
    # plt.show()
    print(y_run)


def main():
    myPerception()
    SKLearnPerception().demo()
    SKLearnMLPClassifier().demo()

if __name__ == '__main__':
    main()
