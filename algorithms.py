import numpy as np

from sklearn.model_selection import StratifiedKFold


def _rbf(d, eps):
    return np.exp(-(d * eps) ** 2)


def _distance(x, y):
    return np.sum(np.abs(x - y))


def _pairwise_distances(X):
    D = np.zeros((len(X), len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue

            d = _distance(X[i], X[j])

            D[i][j] = d
            D[j][i] = d

    return D


def _score(point, X, y, minority_class, epsilon):
    mutual_density_score = 0.0

    for i in range(len(X)):
        rbf = _rbf(_distance(point, X[i]), epsilon)

        if y[i] == minority_class:
            mutual_density_score -= rbf
        else:
            mutual_density_score += rbf

    return mutual_density_score


class RBO:
    def __init__(self, gamma=0.05, n_steps=500, step_size=0.001, stop_probability=0.02, criterion='balance',
                 minority_class=None, n=None):
        assert criterion in ['balance', 'minimize', 'maximize']
        assert 0.0 <= stop_probability <= 1.0

        self.gamma = gamma
        self.n_steps = n_steps
        self.step_size = step_size
        self.stop_probability = stop_probability
        self.criterion = criterion
        self.minority_class = minority_class
        self.n = n

    def fit_sample(self, X, y):
        epsilon = 1.0 / self.gamma
        classes = np.unique(y)

        if self.minority_class is None:
            sizes = [sum(y == c) for c in classes]
            minority_class = classes[np.argmin(sizes)]
        else:
            minority_class = self.minority_class

        minority_points = X[y == minority_class]

        if self.n is None:
            n = sum(y != minority_class) - sum(y == minority_class)
        else:
            n = self.n

        if n == 0:
            return X, y

        minority_scores = []

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            minority_scores.append(_score(minority_point, X, y, minority_class, epsilon))

        appended = []

        while len(appended) < n:
            idx = np.random.choice(range(len(minority_points)))
            point = minority_points[idx].copy()
            score = minority_scores[idx]

            for i in range(self.n_steps):
                if self.stop_probability is not None and self.stop_probability > np.random.rand():
                    break

                translation = np.zeros(len(point))
                sign = np.random.choice([-1, 1])
                translation[np.random.choice(range(len(point)))] = sign * self.step_size
                translated_point = point + translation
                translated_score = _score(translated_point, X, y, minority_class, epsilon)

                if (self.criterion == 'balance' and np.abs(translated_score) < np.abs(score)) or \
                        (self.criterion == 'minimize' and translated_score < score) or \
                        (self.criterion == 'maximize' and translated_score > score):
                    point = translated_point
                    score = translated_score

            appended.append(point)

        return np.concatenate([X, appended]), np.concatenate([y, minority_class * np.ones(len(appended))])


class RBOSelection:
    def __init__(self, classifier, measure, n_splits=5, gammas=(0.05,), n_steps=500, step_size=0.001,
                 stop_probability=0.02, criterion='balance', minority_class=None, n=None):
        self.classifier = classifier
        self.measure = measure
        self.n_splits = n_splits
        self.gammas = gammas
        self.n_steps = n_steps
        self.step_size = step_size
        self.stop_probability = stop_probability
        self.criterion = criterion
        self.minority_class = minority_class
        self.n = n
        self.selected_gamma = None
        self.skf = StratifiedKFold(n_splits=n_splits)

    def fit_sample(self, X, y):
        self.skf.get_n_splits(X, y)

        best_score = -np.inf

        for gamma in self.gammas:
            scores = []

            for train_idx, test_idx in self.skf.split(X, y):
                X_train, y_train = RBO(gamma=gamma, n_steps=self.n_steps, step_size=self.step_size,
                                       stop_probability=self.stop_probability, criterion=self.criterion,
                                       minority_class=self.minority_class, n=self.n).\
                    fit_sample(X[train_idx], y[train_idx])

                classifier = self.classifier.fit(X_train, y_train)
                predictions = classifier.predict(X[test_idx])
                scores.append(self.measure(y[test_idx], predictions))

            score = np.mean(scores)

            if score > best_score:
                self.selected_gamma = gamma

                best_score = score

        return RBO(gamma=self.selected_gamma, n_steps=self.n_steps, step_size=self.step_size,
                   stop_probability=self.stop_probability, criterion=self.criterion,
                   minority_class=self.minority_class, n=self.n).fit_sample(X, y)
