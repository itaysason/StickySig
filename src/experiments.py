from src.models.StickySig.StickySig import StickySig
from src.utils import get_data
import numpy as np
import time


def split_train_test_loco(data, chromosome):
    all_chromosomes = list(data[list(data.keys())[0]].keys())
    if chromosome >= len(all_chromosomes):
        raise ValueError('chromosome={} but there are total of {} chromosomes'.format(chromosome, len(all_chromosomes)))
    out_chromosome = all_chromosomes[chromosome]
    train_data = {}
    test_data = {}
    for sample in data.keys():
        train_data[sample] = {chrom: data[sample][chrom] for chrom in all_chromosomes if chrom != out_chromosome}
        test_data[sample] = {chrom: data[sample][chrom] for chrom in all_chromosomes if chrom == out_chromosome}
    return train_data, test_data


def split_train_test_sample_cv(data, num_folds, fold, shuffle_seed=None):
    if fold >= num_folds:
        raise ValueError('fold={} but there are total of {} folds'.format(fold, num_folds))

    samples = np.array(list(data.keys()))
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
        np.random.shuffle(samples)

    splits = np.array_split(samples, num_folds)
    train_samples = []
    test_samples = []
    for chunk in range(num_folds):
        if chunk == fold:
            test_samples.extend(splits[chunk])
        else:
            train_samples.extend(splits[chunk])
    train_data = {sample: data[sample] for sample in train_samples}
    test_data = {sample: data[sample] for sample in test_samples}
    return train_data, test_data


def train_stickysig(train_data, num_signatures=None, signatures=None, random_seed=None, epsilon=1e-8, max_iterations=200):
    random_seed = time.time() if random_seed is None else random_seed
    np.random.seed(random_seed)
    num_signatures = len(signatures) if signatures is not None else num_signatures
    if num_signatures is None:
        raise ValueError('Both signatures and num_signatures are None')
    model = StickySig(num_signatures)
    if signatures is not None:
        model.e = signatures
        train_ll = model.fit(train_data, learn_params=['pi', 'alpha'], epsilon=epsilon, max_iterations=max_iterations)
    else:
        train_ll = model.fit(train_data, epsilon=epsilon, max_iterations=max_iterations)
    return model, train_ll


def train_test_stickysig(train_data, test_data, num_signatures=None, signatures=None, random_seed=None, epsilon=1e-8, max_iterations=200):
    model, train_ll = train_stickysig(train_data, num_signatures, signatures, random_seed, epsilon, max_iterations)
    refit_data = {key: value for key, value in test_data.items() if key not in model.pi.keys()}
    non_refit_data = {key: value for key, value in test_data.items() if key in model.pi.keys()}
    test_ll = model.refit(refit_data, epsilon, max_iterations)
    test_non_refit_ll = model.log_probability(non_refit_data)
    for sample, value in test_non_refit_ll.items():
        test_ll[sample] = value
    return model, train_ll, test_ll


def get_data_by_model_name(dataset, model_name, use_kataegis=True):
    """
    Possible model names: MM, StickySig, StickySig-same-allele, StickySig-same-strand, StickySig-same-allele-same-strand
    :param dataset:
    :param model_name:
    :param use_kataegis:
    :return:
    """
    if model_name == 'MM':
        strand_name = None
    elif model_name == 'StickySig':
        strand_name = 'no-strand'
    elif model_name == 'StickySig-same-allele':
        strand_name = 'same-allele'
    elif model_name == 'StickySig-same-strand':
        strand_name = 'same-strand'
    elif model_name == 'StickySig-leading':
        strand_name = 'leading'
    elif model_name == 'StickySig-lagging':
        strand_name = 'lagging'
    elif model_name == 'StickySig-leading1-lagging1':
        data_leading, active_signatures = get_data(dataset, 'leading', use_kataegis)
        data_lagging, active_signatures = get_data(dataset, 'lagging', use_kataegis)
        for sample in data_leading.keys():
            sample_keys = data_leading[sample].keys()
            for key in sample_keys:
                data_leading[sample][key]['StrandInfo'] += data_lagging[sample][key]['StrandInfo']
        return data_leading, active_signatures
    elif model_name == 'StickySig-leading1-lagging2':
        data_leading, active_signatures = get_data(dataset, 'leading', use_kataegis)
        data_lagging, active_signatures = get_data(dataset, 'lagging', use_kataegis)
        for sample in data_leading.keys():
            sample_keys = data_leading[sample].keys()
            for key in sample_keys:
                data_leading[sample][key]['StrandInfo'] += 2 * data_lagging[sample][key]['StrandInfo']
        return data_leading, active_signatures
    elif model_name == 'StickySig-same-allele-same-strand':
        data_same_allele, active_signatures = get_data(dataset, 'same-allele', use_kataegis)
        data_same_strand, active_signatures = get_data(dataset, 'same-strand', use_kataegis)
        for sample in data_same_allele.keys():
            sample_keys = data_same_allele[sample].keys()
            for key in sample_keys:
                data_same_allele[sample][key]['StrandInfo'] += data_same_strand[sample][key]['StrandInfo']
        return data_same_allele, active_signatures
    else:
        raise ValueError('{} is not a valid model_name'.format(model_name))
    return get_data(dataset, strand_name, use_kataegis)


def predict_hidden_variables(dataset, parameters, algorithm='viterbi'):
    e = np.array(parameters['e'])
    alpha = np.array(parameters['alpha'])
    pi = {sample: np.array(sample_pi) for sample, sample_pi in parameters['pi'].items()}
    k, m = e.shape
    r = len(alpha)
    model = StickySig(k)
    model.m, model.r = m, r
    model.pi = pi
    model.e = e
    model.alpha = alpha
    prediction = model.predict(dataset, algorithm)
    return prediction


def prepare_data_to_json(data):
    prediction = data

    list_prediction = {}
    for sample, sample_dict in prediction.items():
        list_prediction[sample] = {}
        for chrom, chrom_dict in sample_dict.items():
            list_prediction[sample][chrom] = {}
            list_prediction[sample][chrom]['signatures'] = np.array(chrom_dict['signatures']).tolist()
            list_prediction[sample][chrom]['stickiness'] = np.array(chrom_dict['stickiness']).tolist()
            list_prediction[sample][chrom]['log-likelihood'] = np.array(chrom_dict['log-likelihood']).tolist()

    return list_prediction
