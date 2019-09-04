import json
import numpy as np
import pandas as pd


def save_json(file_name, dict_to_save):
    if 'json' not in file_name:
        file_name += '.json'
    with open(file_name, 'w') as fp:
        json.dump(dict_to_save, fp)


def load_json(file_name):
    if 'json' not in file_name:
        file_name += '.json'
    return json.load(open(file_name))


def get_data(dataset, strand_name=None, use_kataegis=True):
    """
    return the samples numbered in [start, finish) in the dataset parsed to chromosomes.
    The dict returned (dataset_dict) has the samples as keys.
    dataset_dict[sample] is a dictionary with 24: keys 1-22, X, Y, i.e the chromosomes.
    dataset_dict[sample][chromosome] is a dictionary with 4 keys: Sequence, PrevMutDists, sameAlleleAsPrev, statistics
    dataset_dict[sample][chromosome][Sequence] is a numpy array of mutations (ints)
    dataset_dict[sample][chromosome][StrandInfo] is a numpy array of strand continuation indicators (ints)
    :param dataset:
    :param strand_name:
    :param use_kataegis:
    :return:
    """
    if dataset == 'ICGC-BRCA':
        # dataset_path = 'data/mutations/ICGC/BRCA-EU/ICGC-BRCA-EU.RELEASE_22.SBS.renamed.sigma.json'
        dataset_path = 'data/mutations/ICGC/BRCA-EU/ICGC-BRCA-EU.RELEASE_22.SBS.renamed.strand.kataegis.smm.json'
        active_signatures = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    elif dataset == 'PCAWG-CLLE':
        dataset_path = 'data/mutations/PCAWG/CLLE-ES/PCAWG-CLLE-ES-R27.SBS.strand.kataegis.smmm.json'
        active_signatures = [1, 2, 5, 9, 13]
    elif dataset == 'PCAWG-MALY':
        dataset_path = 'data/mutations/PCAWG/MALY-DE/PCAWG-MALY-DE-R27.SBS.strand.kataegis.smmm.json'
        active_signatures = [1, 2, 5, 9, 13, 17]
    elif dataset == 'PCAWG-COAD':
        dataset_path = 'data/mutations/PCAWG/COAD-US/PCAWG-COAD-US-R27.SBS.strand.kataegis.smmm.json'
        active_signatures = [1, 5, 6, 10]
    else:
        raise ValueError('Dataset is not supported')

    # converting active signatures to indices
    active_signatures = np.array(active_signatures) - 1

    # parsing data
    full_dataset = load_json(dataset_path)
    samples = np.sort(full_dataset['samples'])

    chromosome_names = ['%s' % str(i+1) for i in range(22)]
    chromosome_names.extend(['X', 'Y'])

    if strand_name == 'no-strand':
        strand_key = 'No strand'
    elif strand_name == 'same-allele':
        strand_key = 'Same allele'
    elif strand_name == 'same-strand':
        strand_key = 'Same strand'
    elif strand_name == 'leading':
        strand_key = 'Leading strand'
    elif strand_name == 'lagging':
        strand_key = 'Lagging strand'
    elif strand_name == 'template':
        strand_key = 'Template strand'
    elif strand_name == 'non-template':
        strand_key = 'Non-template strand'
    elif strand_name == 'same-direction':
        strand_key = 'Replication/Transcription Same direction'
    elif strand_name == 'opposite-direction':
        strand_key = 'Replication/Transcription Opposite direction'
    elif strand_name is None:
        strand_key = None
    else:
        raise ValueError('{} is not a valid strand name.'.format(strand_name))

    sample_to_sequences = full_dataset['sampleToSequence']
    sample_to_chromosome_map = full_dataset['chromosomeMap']
    sample_to_strand_info = full_dataset['sampleToStrandMatch']
    sample_to_kataegis = full_dataset['sampleToKataegis']

    dataset_dict = {}
    for sample in samples:
        sample_dict = {}
        sequence = np.array(sample_to_sequences[sample], dtype='int')
        if strand_key is None:
            strand_info = np.zeros(len(sequence), dtype='int')
        else:
            strand_info = np.array(sample_to_strand_info[sample][strand_key], dtype='int')
        chromosome_map = np.array(sample_to_chromosome_map[sample])
        kataegis = np.array(sample_to_kataegis[sample], dtype='int')
        for chromosome in chromosome_names:
            indices = np.where(chromosome_map == chromosome)[0]
            curr_strand_info = strand_info[indices]
            if not use_kataegis:
                kataegis_indices = np.where(kataegis[indices] == 1)[0]
                curr_strand_info[kataegis_indices] = 0
            sample_dict[chromosome] = {'Sequence': sequence[indices], 'StrandInfo': curr_strand_info}

        dataset_dict[sample] = sample_dict

    return dataset_dict, active_signatures


def get_cosmic_signatures(dir_path='data/signatures/COSMIC/cosmic-signatures.tsv'):
    return pd.read_csv(dir_path, sep='\t', index_col=0).values
