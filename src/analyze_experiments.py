import os
from src.utils import load_json
import numpy as np


def process_loco(loco_dir='experiments/LOCO'):
    chromosomes = [str(i) for i in range(1, 23)]
    chromosomes.extend(['X', 'Y'])
    chromosomes = np.array(chromosomes)

    experiment_string = '{} with {} signatures'

    datasets = os.listdir(loco_dir)
    for dataset in datasets:
        print(dataset + ':')
        dataset_dir = os.path.join(loco_dir, dataset)
        for signature_learning in os.listdir(dataset_dir):
            for model in os.listdir(os.path.join(dataset_dir, signature_learning)):
                for num_sigs in os.listdir(os.path.join(dataset_dir, signature_learning, model)):
                    if signature_learning == 'denovo':
                        signatures_string = str(num_sigs)
                    else:
                        signatures_string = '{} known cosmic'.format(num_sigs)
                    curr_experiment_string = experiment_string.format(model, signatures_string)
                    experiment_dir = os.path.join(dataset_dir, signature_learning, model, num_sigs)
                    folds = np.array(os.listdir(experiment_dir))
                    no_dir_folds = [fold for fold in chromosomes if fold not in folds]
                    no_run_folds = []
                    experiment_score = 0
                    for fold in folds:
                        runs = os.listdir(os.path.join(experiment_dir, fold))
                        num_runs = len(runs)
                        if num_runs == 0:
                            no_run_folds.append(fold)
                            continue
                        train_scores = np.zeros(num_runs)
                        test_scores = np.zeros(num_runs)
                        for i, run in enumerate(runs):
                            file_path = os.path.join(experiment_dir, fold, run)
                            run_dict = load_json(file_path)
                            for s in run_dict['log-likelihood-train'].values():
                                train_scores[i] += s
                            for s in run_dict['log-likelihood-test'].values():
                                test_scores[i] += s

                        # Deciding what run to use according to the log likelihood of the train data
                        best_run = np.argmax(train_scores)
                        experiment_score += test_scores[best_run]
                    if len(no_run_folds) > 0 or len(no_dir_folds):
                        print(curr_experiment_string + ' completely missing folds {} and missing runs for {}'.format(no_dir_folds, no_run_folds))
                    else:
                        print(curr_experiment_string + ' score is {}'.format(experiment_score))
        print('\n')


def process_sample_cv(sample_cv_dir='experiments/sampleCV'):
    experiment_string = '{}fold CV on {} with {} signatures'

    datasets = os.listdir(sample_cv_dir)
    for dataset in datasets:
        print(dataset + ':')
        dataset_dir = os.path.join(sample_cv_dir, dataset)
        for signature_learning in os.listdir(dataset_dir):
            for model in os.listdir(os.path.join(dataset_dir, signature_learning)):
                for num_sigs in os.listdir(os.path.join(dataset_dir, signature_learning, model)):
                    if signature_learning == 'denovo':
                        signatures_string = str(num_sigs)
                    else:
                        signatures_string = '{} known cosmic'.format(num_sigs)
                    # curr_experiment_string = experiment_string.format(model, kataegis_string, signatures_string)
                    experiment_dir = os.path.join(dataset_dir, signature_learning, model, num_sigs)
                    shuffle_seed = os.listdir(experiment_dir)[0]
                    experiment_dir = os.path.join(experiment_dir, shuffle_seed)
                    total_folds = np.array(os.listdir(experiment_dir))
                    for num_folds in total_folds:
                        curr_experiment_string = experiment_string.format(num_folds, model, signatures_string)
                        all_folds = [str(i) for i in range(int(num_folds))]
                        cv_dir = os.path.join(experiment_dir, num_folds)
                        folds = os.listdir(cv_dir)
                        no_dir_folds = [fold for fold in all_folds if fold not in folds]
                        no_run_folds = []
                        experiment_score = 0
                        for fold in folds:
                            runs = os.listdir(os.path.join(cv_dir, fold))
                            num_runs = len(runs)
                            if num_runs == 0:
                                no_run_folds.append(fold)
                                continue
                            train_scores = np.zeros(num_runs)
                            test_scores = np.zeros(num_runs)
                            for i, run in enumerate(runs):
                                file_path = os.path.join(cv_dir, fold, run)
                                run_dict = load_json(file_path)
                                for s in run_dict['log-likelihood-train'].values():
                                    train_scores[i] += s
                                for s in run_dict['log-likelihood-test'].values():
                                    test_scores[i] += s

                            # Deciding what run to use according to the log likelihood of the train data
                            best_run = np.argmax(train_scores)
                            experiment_score += test_scores[best_run]
                        if len(no_run_folds) > 0 or len(no_dir_folds):
                            print(curr_experiment_string + 'completely missing folds {} and missing runs for {}'.format(no_dir_folds, no_run_folds))
                        else:
                            print(curr_experiment_string + ' score is {}'.format(experiment_score))
        print('\n')


def process_trained_models(trained_models_dir='experiments/trained_models'):
    datasets = os.listdir(trained_models_dir)
    for dataset in datasets:
        print(dataset + ':')
        dataset_dir = os.path.join(trained_models_dir, dataset)
        for signature_learning in os.listdir(dataset_dir):
            for model in os.listdir(os.path.join(dataset_dir, signature_learning)):
                for num_sigs in os.listdir(os.path.join(dataset_dir, signature_learning, model)):
                    experiment_dir = os.path.join(dataset_dir, signature_learning, model, num_sigs)
                    runs = os.listdir(experiment_dir)
                    print(experiment_dir)
                    for run in runs:
                        scores = load_json(os.path.join(experiment_dir, run))['log-likelihood']
                        total_score = 0
                        for score in scores.values():
                            total_score += score
                        print(run, total_score)
        print('\n')
