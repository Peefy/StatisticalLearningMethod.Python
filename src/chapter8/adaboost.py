
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class SKLearnAdaboost:
    def __init__(self):
        pass

    def demo(self):
        h = .02
        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes", "QDA"]
        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]
        X ,y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
        linearly_separable = (X, y)
        datasets = [make_moons(noise=0.3, random_state=0),
                    make_circles(noise=0.2, factor=0.5, random_state=1),
                    linearly_separable]
        figure = plt.figure(figsize=(27, 9))
        i = 1
        # iterable over datasets
        for ds_cnt, ds in enumerate(datasets):
            # preprocess dataset, split into training and test part
            X, y = ds
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=.4, random_state=42)
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            # just plot the dataset first
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(['#FF0000', '#0000FF'])
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            if ds_cnt == 0:
                ax.set_title("Input data")
            # Plot the training points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                           edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            i += 1

            # iterate over classifiers
            for name, clf in zip(names, classifiers):
                ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)

                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                if hasattr(clf, "decision_function"):
                    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                else:
                    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

                # Plot the training points
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                           edgecolors='k')
                # Plot the testing points
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                           edgecolors='k', alpha=0.6)

                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                if ds_cnt == 0:
                    ax.set_title(name)
                ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                        size=15, horizontalalignment='right')
                i += 1

        plt.show()
        return self

    def twoClassDemo(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import make_gaussian_quantiles

        # Construct dataset
        X1, y1 = make_gaussian_quantiles(cov=2.,
                                         n_samples=200, n_features=2,
                                         n_classes=2, random_state=1)
        X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                         n_samples=300, n_features=2,
                                         n_classes=2, random_state=1)
        X = np.concatenate((X1, X2))
        y = np.concatenate((y1, - y2 + 1))

        # Create and fit an AdaBoosted decision tree
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                 algorithm="SAMME",
                                 n_estimators=200)

        bdt.fit(X, y)

        plot_colors = "br"
        plot_step = 0.02
        class_names = "AB"

        plt.figure(figsize=(10, 5))

        # Plot the decision boundaries
        plt.subplot(121)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        plt.axis("tight")

        # Plot the training points
        for i, n, c in zip(range(2), class_names, plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1],
                        c=c, cmap=plt.cm.Paired,
                        s=20, edgecolor='k',
                        label="Class %s" % n)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend(loc='upper right')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Decision Boundary')

        # Plot the two-class decision scores
        twoclass_output = bdt.decision_function(X)
        plot_range = (twoclass_output.min(), twoclass_output.max())
        plt.subplot(122)
        for i, n, c in zip(range(2), class_names, plot_colors):
            plt.hist(twoclass_output[y == i],
                     bins=10,
                     range=plot_range,
                     facecolor=c,
                     label='Class %s' % n,
                     alpha=.5,
                     edgecolor='k')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, y1, y2 * 1.2))
        plt.legend(loc='upper right')
        plt.ylabel('Samples')
        plt.xlabel('Score')
        plt.title('Decision Scores')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        plt.show()

        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import datasets
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import zero_one_loss
        from sklearn.ensemble import AdaBoostClassifier

        n_estimators = 400
        # A learning rate of 1. may not be optimal for both SAMME and SAMME.R
        learning_rate = 1.

        X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)

        X_test, y_test = X[2000:], y[2000:]
        X_train, y_train = X[:2000], y[:2000]

        dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
        dt_stump.fit(X_train, y_train)
        dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

        dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
        dt.fit(X_train, y_train)
        dt_err = 1.0 - dt.score(X_test, y_test)

        ada_discrete = AdaBoostClassifier(
            base_estimator=dt_stump,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            algorithm="SAMME")
        ada_discrete.fit(X_train, y_train)

        ada_real = AdaBoostClassifier(
            base_estimator=dt_stump,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            algorithm="SAMME.R")
        ada_real.fit(X_train, y_train)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
                label='Decision Stump Error')
        ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
                label='Decision Tree Error')

        ada_discrete_err = np.zeros((n_estimators,))
        for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
            ada_discrete_err[i] = zero_one_loss(y_pred, y_test)

        ada_discrete_err_train = np.zeros((n_estimators,))
        for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
            ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

        ada_real_err = np.zeros((n_estimators,))
        for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
            ada_real_err[i] = zero_one_loss(y_pred, y_test)

        ada_real_err_train = np.zeros((n_estimators,))
        for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
            ada_real_err_train[i] = zero_one_loss(y_pred, y_train)

        ax.plot(np.arange(n_estimators) + 1, ada_discrete_err,
                label='Discrete AdaBoost Test Error',
                color='red')
        ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train,
                label='Discrete AdaBoost Train Error',
                color='blue')
        ax.plot(np.arange(n_estimators) + 1, ada_real_err,
                label='Real AdaBoost Test Error',
                color='orange')
        ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
                label='Real AdaBoost Train Error',
                color='green')

        ax.set_ylim((0.0, 0.5))
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('error rate')

        leg = ax.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.7)

        plt.show()


def main():
    SKLearnAdaboost().demo().twoClassDemo()


if __name__ == "__main__":
    main()
