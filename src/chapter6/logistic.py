
class SKLearnLogisticRegression:
    def __init__(self):
        pass

    def demo(self):
        import numpy as np
        import matplotlib.pyplot as plt

        plt.figure()
        plt.axis([-6, 6, 0, 1])
        plt.grid(True)
        X = np.arange(-6, 6, 0.1)
        y = 1 / (1 + np.e ** (-X))
        plt.plot(X, y, 'b-')
        plt.show()

        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression

        diabetes = datasets.load_iris()

        X_train, X_test, y_train, y_test = train_test_split(diabetes.data,
                                                            diabetes.target, test_size=0.30,random_state=0)

        cls = LogisticRegression(multi_class='multinomial',solver='lbfgs')

        cls.fit(X_train, y_train)

        print("Coefficients:%s, intercept %s" % (cls.coef_, cls.intercept_))
        print("Residual sum of squares: %.2f" % np.mean((cls.predict(X_test) - y_test) ** 2))
        print('Score: %.2f' % cls.score(X_test, y_test))


def main():
    SKLearnLogisticRegression().demo()


if __name__ == '__main__':
    main()
