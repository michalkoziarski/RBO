import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import databases
import datasets


for dataset in datasets.names('preliminary'):
    for fold in range(1, 11):
        for classifier in ['KNN', 'CART', 'SVM', 'NB']:
            for n_steps in [1000, 2000, 4000, 8000, 16000]:
                for gamma in [0.001, 0.01, 0.1, 1.0, 10.0]:
                    trial = {
                        'Algorithm': 'RBO',
                        'Parameters': {
                            'gamma': gamma,
                            'n_steps': n_steps,
                            'step_size': 0.001,
                            'stop_probability': 0.0,
                            'criterion': 'balance',
                            'classifier': classifier
                        },
                        'Dataset': dataset,
                        'Fold': fold,
                        'Description': 'Preliminary'
                    }

                    databases.add_to_pending(trial)
