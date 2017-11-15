import numpy as np
import multiprocessing as mp

from algorithms import RBO, RBOSelection
from databases import pull_pending, submit_result
from datasets import load

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as CART
from sklearn.svm import SVC as SVM
from sklearn.naive_bayes import GaussianNB as NB
from imblearn.under_sampling import NeighbourhoodCleaningRule as NCL
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


N_PROCESSES = 24


def run():
    while True:
        trial = pull_pending()

        if trial is None:
            break

        params = eval(trial['Parameters'])

        print(trial)

        clf = eval(params['classifier'])()

        if trial['Algorithm'] == 'RBO':
            algorithm = RBO(gamma=params['gamma'], n_steps=params['n_steps'], step_size=params['step_size'],
                            stop_probability=params['stop_probability'], criterion=params['criterion'])
        elif trial['Algorithm'] == 'RBOSelection':
            if params['measure'] == 'AUC':
                measure = metrics.roc_auc_score
            else:
                raise NotImplementedError

            algorithm = RBOSelection(classifier=clf, measure=measure, gammas=params['gammas'], n_steps=params['n_steps'],
                                     step_size=params['step_size'], stop_probability=params['stop_probability'],
                                     criterion=params['criterion'])
        elif (trial['Algorithm'] is None) or (trial['Algorithm'] == 'None'):
            algorithm = None
        else:
            algorithms = {
                'SMOTE': SMOTE(),
                'SMOTE+ENN': SMOTEENN(),
                'SMOTE+TL': SMOTETomek(),
                'Bord': SMOTE(kind='borderline1'),
                'ADASYN': ADASYN(),
                'NCL': NCL()
            }

            algorithm = algorithms.get(trial['Algorithm'])

            if algorithm is None:
                raise NotImplementedError

        dataset = load(trial['Dataset'], noise_type=params.get('noise_type', None),
                       noise_level=params.get('noise_level', 0.0))
        fold = int(trial['Fold']) - 1

        (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

        labels = np.unique(y_test)
        counts = [len(y_test[y_test == label]) for label in labels]
        minority_class = labels[np.argmin(counts)]

        if algorithm.__class__ in [SMOTE, SMOTEENN, SMOTETomek]:
            train_labels = np.unique(y_train)
            train_counts = [len(y_train[y_train == train_label]) for train_label in train_labels]
            train_minority_class = labels[np.argmin(train_counts)]
            algorithm.k = algorithm.k_neighbors = np.min([len(y_train[y_train == train_minority_class]) - 1, 5])

        if algorithm is not None:
            X_train, y_train = algorithm.fit_sample(X_train, y_train)

        clf = clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        g_mean = 1.0

        for label in np.unique(y_test):
            idx = (y_test == label)
            g_mean *= metrics.accuracy_score(y_test[idx], predictions[idx])

        g_mean = np.sqrt(g_mean)

        scores = {
            'Accuracy': metrics.accuracy_score(y_test, predictions),
            'Average accuracy': np.mean(metrics.recall_score(y_test, predictions, average=None)),
            'Precision': metrics.precision_score(y_test, predictions, pos_label=minority_class),
            'Recall': metrics.recall_score(y_test, predictions, pos_label=minority_class),
            'F-measure': metrics.f1_score(y_test, predictions, pos_label=minority_class),
            'AUC': metrics.roc_auc_score(y_test, predictions),
            'G-mean': g_mean
        }

        submit_result(trial, scores)


if __name__ == '__main__':
    for _ in range(N_PROCESSES):
        mp.Process(target=run).start()
