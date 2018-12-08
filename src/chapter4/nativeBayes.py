
class SKLearnNativeBayes:
    def __init__(self):
        pass

    def demo(self):
        import numpy as np
        from sklearn.naive_bayes import GaussianNB
        X = np.array([[-1, -1], [-2, -2], [-3, -3],[-4,-4],[-5,-5], [1, 1], [2, 2], [3, 3]])
        y = np.array([1, 1, 1, 1, 1, 2, 2, 2])
        clf = GaussianNB()
        clf.fit(X, y)
        print(clf.theta_)
        print(clf.sigma_)
        print(clf.predict([[-6, -6], [4, 5]]))
        print(clf.predict_proba([[-6, -6], [4, 5]]))
        print(clf.score([[-6, -6], [-4, -2], [-3, -4], [4, 5]], [1, 1, 2, 2]))

        from sklearn.naive_bayes import MultinomialNB
        X = np.array([[1, 2, 3, 4], [1, 3, 4, 4], [2, 4, 5, 5], [2, 5, 6, 5], [3, 4, 5, 6], [3, 5, 6, 6]])
        y = np.array([1, 1, 4, 2, 3, 3])
        clf = MultinomialNB(alpha=2.0)
        clf.fit(X, y)
        print(clf.class_log_prior_)
        print(clf.intercept_)
        print(clf.feature_log_prob_)
        print(clf.coef_)
        print(clf.class_count_)
        print(np.log(2 / 6), np.log(1 / 6), np.log(2 / 6), np.log(1 / 6))
        print(clf.predict([[1, 3, 5, 6], [3, 4, 5, 4]]))
        print(clf.score([[3, 4, 5, 4], [1, 3, 5, 6]], [1, 1]))

        from sklearn.naive_bayes import BernoulliNB
        X = np.array([[1, 2, 3, 4], [1, 3, 4, 4], [2, 4, 5, 5]])
        y = np.array([1, 1, 2])
        clf = BernoulliNB(alpha=2.0, binarize=3.0, fit_prior=True)
        clf.fit(X, y)
        print(clf.class_log_prior_)
        print(clf.intercept_)
        print(clf.feature_log_prob_)
        print(clf.coef_)
        print(clf.class_count_)
        print(clf.predict([[1, 3, 5, 6], [3, 4, 5, 4]]))
        print(clf.score([[3, 4, 5, 4], [1, 3, 5, 6]], [1, 1]))


def main():
    SKLearnNativeBayes().demo()


if __name__ == "__main__":
    main()
