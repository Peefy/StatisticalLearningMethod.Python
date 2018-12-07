import csv
import random


class BTreeNode:
    """
    二叉树结点
    """

    def __init__(self, left, right, index, key, leftindex, rightindex):
        """
        二叉树结点

        Args
        ===
        `left` : BTreeNode : 左儿子结点

        `right`  : BTreeNode : 右儿子结点

        `index` : 结点自身索引值

        `key` : 结点自身键值

        `leftindex` : 左儿子结点索引值

        `rightindex` : 右儿子结点索引值

        """

        self.leftindex = leftindex
        self.rightindex = rightindex
        self.left = left
        self.right = right
        self.index = index
        self.key = key


class BinaryTree:
    '''
    二叉树
    '''

    def __init__(self):
        '''
        二叉树
        '''
        self.lastnode = None
        self.root = None
        self.nodes = []

    def addnode(self, leftindex: int, rightindex: int, selfindex: int, selfkey):
        '''
        加入二叉树结点

        Args
        ===
        `leftindex` : 左儿子结点索引值

        `rightindex` : 右儿子结点索引值

        `selfindex` : 结点自身索引值

        `selfkey` : 结点自身键值

        '''
        leftnode = self.findnode(leftindex)
        rightnode = self.findnode(rightindex)
        x = BTreeNode(leftnode, rightnode, selfindex, \
                      selfkey, leftindex, rightindex)
        self.nodes.append(x)
        self.lastnode = x
        return x

    def renewall(self) -> None:
        '''
        更新/连接/构造二叉树
        '''
        for node in self.nodes:
            node.left = self.findnode(node.leftindex)
            node.right = self.findnode(node.rightindex)

    def findleftrightnode(self, node: BTreeNode) -> list:
        '''
        找出二叉树某结点的所有子结点

        Args
        ===
        `node` : BTreeNode : 某结点
        '''
        array = []
        if node != None:
            # 递归找到左儿子所有的结点
            leftnodes = self.findleftrightnode(node.left)
            # 递归找到右兄弟所有的结点
            rightnodes = self.findleftrightnode(node.right)
            if leftnodes != None and len(leftnodes) != 0:
                # 连接两个集合
                array = array + leftnodes
            if rightnodes != None and len(rightnodes) != 0:
                # 连接两个集合
                array = array + rightnodes
            # 将自己本身的结点也加入集合
            array.append({"index": node.index, "key": node.key})
            if len(array) == 0:
                return None
            return array
        return None

    def all(self) -> list:
        '''
        返回二叉树中所有结点索引值，键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append({"index": node.index, "key": node.key})
        return array

    def keys(self) -> list:
        '''
        返回二叉树中所有结点键值构成的集合
        '''
        array = []
        for node in self.nodes:
            array.append(node.key)
        return array

    def findnode(self, index: int):
        '''
        根据索引寻找结点`O(n)`

        Args
        ===
        `index` : 索引值
        '''
        if index == None:
            return None
        for node in self.nodes:
            if node.index == index:
                return node
        return None


class KDTreeNode:
    pass


class KDTree(BinaryTree):
    """
    kd二叉树
    """
    pass


class SKLearnKNeighborsClassifier:
    """
    K近邻算法的sklearn算法实现
    """

    def demo(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets.samples_generator import make_classification
        from sklearn.neighbors import KNeighborsClassifier
        # X为样本特征，Y为样本类别输出， 共1000个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
        X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, \
                                   n_clusters_per_class=1, n_classes=3)

        # plot出样本集合的二维x,y坐标
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
        plt.title('生成数据二维坐标和类别图', fontproperties="SimHei")

        # 划分训练集、测试集：
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, y)

        clf = KNeighborsClassifier(n_neighbors=15, weights='distance')
        clf.fit(X, y)

        from matplotlib.colors import ListedColormap
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # 确认训练集的边界
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        # 生成随机数据来做测试集，然后作预测
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # 画出测试集数据
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # 也画出所有的训练集数据
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3st-Class classification (k = 15, weights = 'diance')")
        plt.show()

    def loadDataset(self, filename, split, trainSet=None, testSet=[]):
        if trainSet is None:
            trainSet = []
        with open(filename, 'rb') as csvfile:
            lines = csv.reader(csvfile)
            dataset = list(lines)
            for x in range(len(dataset) - 1):
                for y in range(4):
                    dataset[x][y] = float(dataset[x][y])
                if random.random() < split:
                    trainSet.append(dataset[x])
                else:
                    testSet.append(dataset[y])


class SKlearnKNeighborsRegressor:
    def demo(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import neighbors
        np.random.seed(0)
        ndatanum = 100
        X = np.sort(5 * np.random.rand(100, 1), axis=0)
        T = np.linspace(0, 5, 500)[:, np.newaxis]
        y = np.sin(X).ravel()

        # Add noise to targets
        y[::5] += 1 * (0.5 * np.random.rand(20))

        # Fit regression model
        n_neighbors = 5
        for i, weights in enumerate(['uniform', 'distance']):
            knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
            y_ = knn.fit(X, y).predict(T)

            plt.subplot(2, 1, i + 1)
            plt.scatter(X, y, c='k', label='data')
            plt.plot(T, y_, c='g', label='prediction')
            plt.legend()
            plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

        plt.show()


def main():
    SKLearnKNeighborsClassifier().demo()
    SKlearnKNeighborsRegressor().demo()


if __name__ == '__main__':
    main()
