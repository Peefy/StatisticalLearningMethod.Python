
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler, Nystroem)

from time import time

class SKLearnSVM:
    def __init__(self):
        pass

    def demo(self):
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        y = np.array([1, 1, 2, 2])
        
        clf = SVC(gamma='auto')
        clf.fit(X, y)
        print(clf.predict([[-0.8, 1]]))
        return self

    def multiclassifier_demo(self):
        plt.figure(figsize=(8, 6))
        X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                              allow_unlabeled=True,
                                              random_state=1)
        self.plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
        self.plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")

        X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                              allow_unlabeled=False,
                                              random_state=1)

        self.plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
        self.plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

        plt.subplots_adjust(0.04, 0.02, 0.97, 0.94, 0.09, 0.2)
        plt.show()
        return self

    def plot_hyperplane(self, clf, min_x, max_x, linestyle, label):
        # get the separating hyperplane
        w = clf.coef_[0]
        a = -w[0] / w[1]
        # make sure the line is long enough
        xx = np.linspace(min_x - 5, max_x + 5)
        yy = a * xx - (clf.intercept_[0]) / w[1]
        plt.plot(xx, yy, linestyle, label=label)


    def plot_subfigure(self, X, Y, subplot, title, transform):
        if transform == "pca":
            X = PCA(n_components=2).fit_transform(X)
        elif transform == "cca":
            X = CCA(n_components=2).fit(X, Y).transform(X)
        else:
            raise ValueError

        min_x = np.min(X[:, 0])
        max_x = np.max(X[:, 0])

        min_y = np.min(X[:, 1])
        max_y = np.max(X[:, 1])

        classif = OneVsRestClassifier(SVC(kernel='linear'))
        classif.fit(X, Y)

        plt.subplot(2, 2, subplot)
        plt.title(title)

        zero_class = np.where(Y[:, 0])
        one_class = np.where(Y[:, 1])
        plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
        plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
                    facecolors='none', linewidths=2, label='Class 2')

        self.plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                        'Boundary\nfor class 1')
        self.plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                        'Boundary\nfor class 2')

        plt.xticks()
        plt.yticks()

        plt.xlim(min_x - 0.5 * max_x, max_x + 0.5 * max_x)
        plt.ylim(min_y - 0.5 * max_y, max_y + 0.5 * max_y)
        if subplot == 2:
            plt.xlabel('First principal component')
            plt.ylabel('Second principal component')
            plt.legend(loc="upper left")

    def explicit_map_for_RBF_demo(self):
        # The digits dataset
        digits = datasets.load_digits(n_class=9)

        # To apply an classifier on this data, we need to flatten the image, to
        # turn the data in a (samples, feature) matrix:
        n_samples = len(digits.data)
        data = digits.data / 16.
        data -= data.mean(axis=0)

        # We learn the digits on the first half of the digits
        data_train, targets_train = (data[:n_samples // 2],
                                     digits.target[:n_samples // 2])

        # Now predict the value of the digit on the second half:
        data_test, targets_test = (data[n_samples // 2:],
                                   digits.target[n_samples // 2:])
        # data_test = scaler.transform(data_test)

        # Create a classifier: a support vector classifier
        kernel_svm = svm.SVC(gamma=.2)
        linear_svm = svm.LinearSVC()

        # create pipeline from kernel approximation
        # and linear svm
        feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
        feature_map_nystroem = Nystroem(gamma=.2, random_state=1)
        fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                                ("svm", svm.LinearSVC())])

        nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                                 ("svm", svm.LinearSVC())])

        # fit and predict using linear and kernel svm:

        kernel_svm_time = time()
        kernel_svm.fit(data_train, targets_train)
        kernel_svm_score = kernel_svm.score(data_test, targets_test)
        kernel_svm_time = time() - kernel_svm_time

        linear_svm_time = time()
        linear_svm.fit(data_train, targets_train)
        linear_svm_score = linear_svm.score(data_test, targets_test)
        linear_svm_time = time() - linear_svm_time

        sample_sizes = 30 * np.arange(1, 10)
        fourier_scores = []
        nystroem_scores = []
        fourier_times = []
        nystroem_times = []

        for D in sample_sizes:
            fourier_approx_svm.set_params(feature_map__n_components=D)
            nystroem_approx_svm.set_params(feature_map__n_components=D)
            start = time()
            nystroem_approx_svm.fit(data_train, targets_train)
            nystroem_times.append(time() - start)

            start = time()
            fourier_approx_svm.fit(data_train, targets_train)
            fourier_times.append(time() - start)

            fourier_score = fourier_approx_svm.score(data_test, targets_test)
            nystroem_score = nystroem_approx_svm.score(data_test, targets_test)
            nystroem_scores.append(nystroem_score)
            fourier_scores.append(fourier_score)

        # plot the results:
        plt.figure(figsize=(8, 8))
        accuracy = plt.subplot(211)
        # second y axis for timeings
        timescale = plt.subplot(212)

        accuracy.plot(sample_sizes, nystroem_scores, label="Nystroem approx. kernel")
        timescale.plot(sample_sizes, nystroem_times, '--',
                       label='Nystroem approx. kernel')

        accuracy.plot(sample_sizes, fourier_scores, label="Fourier approx. kernel")
        timescale.plot(sample_sizes, fourier_times, '--',
                       label='Fourier approx. kernel')

        # horizontal lines for exact rbf and linear kernels:
        accuracy.plot([sample_sizes[0], sample_sizes[-1]],
                      [linear_svm_score, linear_svm_score], label="linear svm")
        timescale.plot([sample_sizes[0], sample_sizes[-1]],
                       [linear_svm_time, linear_svm_time], '--', label='linear svm')

        accuracy.plot([sample_sizes[0], sample_sizes[-1]],
                      [kernel_svm_score, kernel_svm_score], label="rbf svm")
        timescale.plot([sample_sizes[0], sample_sizes[-1]],
                       [kernel_svm_time, kernel_svm_time], '--', label='rbf svm')

        # vertical line for dataset dimensionality = 64
        accuracy.plot([64, 64], [0.7, 1], label="n_features")

        # legends and labels
        accuracy.set_title("Classification accuracy")
        timescale.set_title("Training times")
        accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
        accuracy.set_xticks(())
        accuracy.set_ylim(np.min(fourier_scores), 1)
        timescale.set_xlabel("Sampling steps = transformed feature dimension")
        accuracy.set_ylabel("Classification accuracy")
        timescale.set_ylabel("Training time in seconds")
        accuracy.legend(loc='best')
        timescale.legend(loc='best')

        # visualize the decision surface, projected down to the first
        # two principal components of the dataset
        pca = PCA(n_components=8).fit(data_train)

        X = pca.transform(data_train)

        # Generate grid along first two principal components
        multiples = np.arange(-2, 2, 0.1)
        # steps along first component
        first = multiples[:, np.newaxis] * pca.components_[0, :]
        # steps along second component
        second = multiples[:, np.newaxis] * pca.components_[1, :]
        # combine
        grid = first[np.newaxis, :, :] + second[:, np.newaxis, :]
        flat_grid = grid.reshape(-1, data.shape[1])

        # title for the plots
        titles = ['SVC with rbf kernel',
                  'SVC (linear kernel)\n with Fourier rbf feature map\n'
                  'n_components=100',
                  'SVC (linear kernel)\n with Nystroem rbf feature map\n'
                  'n_components=100']

        plt.tight_layout()
        plt.figure(figsize=(12, 5))

        # predict and plot
        for i, clf in enumerate((kernel_svm, nystroem_approx_svm,
                                 fourier_approx_svm)):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            plt.subplot(1, 3, i + 1)
            Z = clf.predict(flat_grid)

            # Put the result into a color plot
            Z = Z.reshape(grid.shape[:-1])
            plt.contourf(multiples, multiples, Z, cmap=plt.cm.Paired)
            plt.axis('off')

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=targets_train, cmap=plt.cm.Paired,
                        edgecolors=(0, 0, 0))

            plt.title(titles[i])
        plt.tight_layout()
        plt.show()
        return self


def main():
    SKLearnSVM().demo().multiclassifier_demo().explicit_map_for_RBF_demo()


if __name__ == '__main__':
    main()
