import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import databases
import datasets


for dataset in datasets.names('final'):
    for fold in range(1, 11):
        for noise_type in [None, 'class', 'attribute']:
            if noise_type == 'class':
                noise_levels = [0.05, 0.1, 0.15, 0.2]
            elif noise_type == 'attribute':
                noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
            else:
                noise_levels = [0.0]

            for noise_level in noise_levels:
                for classifier in ['KNN', 'CART', 'SVM', 'NB']:
                    trial = {
                        'Algorithm': 'RBOSelection',
                        'Parameters': {
                            'gammas': [0.001, 0.01, 0.1, 1.0, 10.0],
                            'n_steps': 5000,
                            'step_size': 0.0001,
                            'stop_probability': 0.001,
                            'criterion': 'balance',
                            'measure': 'AUC',
                            'classifier': classifier,
                            'noise_type': noise_type,
                            'noise_level': noise_level
                        },
                        'Dataset': dataset,
                        'Fold': fold,
                        'Description': 'Final'
                    }

                    databases.add_to_pending(trial)

                    for algorithm in ['None', 'SMOTE', 'SMOTE+ENN', 'SMOTE+TL', 'Bord', 'ADASYN', 'NCL']:
                        trial = {
                            'Algorithm': algorithm,
                            'Parameters': {
                                'classifier': classifier,
                                'noise_type': noise_type,
                                'noise_level': noise_level
                            },
                            'Dataset': dataset,
                            'Fold': fold,
                            'Description': 'Final'
                        }

                        databases.add_to_pending(trial)
