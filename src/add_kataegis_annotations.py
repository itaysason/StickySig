#!/usr/bin/env python

# Load required modules
import sys, os, argparse, json, pandas as pd, logging, numpy as np
from collections import OrderedDict

# Load our modules
from data_utils import get_logger
from constants import *

# Command-line argument parser
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--mutations_file', type=str, required=True)
    parser.add_argument('-of', '--output_file', type=str, required=True)
    parser.add_argument('-v', '--verbosity', type=int, required=False,
                        default=logging.INFO)
    parser.add_argument('-nc', '--n_cores', type=int, required=False,
                        default=1)
    return parser

# Classify kataegis for each mutation, given their intermutation distance
# For each mutation, finds the largest kataegis loci that mutation
# could participate in. Inspired by and based off of:
# https://github.com/ellisjj/GenomicVis/blob/master/R/find.kataegis.helper.R
def find_kataegis_loci(imd, patient, chrom):
    n = len(imd)
    kataegis_classes = np.array([0] * n)
    kataegis_loci = []
    cumsum = np.cumsum(imd)
    for i in range(n):
        kataegis_locus = None
        for j in range(n, i + KATAEGIS_MIN_MUT-1, -1):
            #checking if mutations at positions i..j are kataegis
            n_mutations = j-i
            if n_mutations >= KATAEGIS_MIN_MUT and (cumsum[j-1] - cumsum[i]) / (n_mutations - 1) <= KATAEGIS_IMD:
                kataegis_locus = (i,j)
                kataegis_classes[i:j] = 1
                break

        kataegis_loci.append( kataegis_locus )

    return kataegis_classes.tolist(), patient, chrom

# Main
def run( args ):
    # Set up logger
    logger = get_logger(args.verbosity)
    logger.info('[Loading input data]')

    # Load mutations
    mut_df = pd.read_csv(args.mutations_file, sep='\t')
    samples = list(OrderedDict.fromkeys(mut_df[PATIENT]))
    sample_index = dict(zip(samples, range(len(samples))))
    logger.info('- Loaded %s mutations from %s samples' % (len(mut_df), len(samples)))

    # Set up multiprocessing
    logger.info('[Classifying kataegis (%s+ mutations and IMD<=%s)]' % (KATAEGIS_MIN_MUT, KATAEGIS_IMD))
    from sklearn.externals.joblib import Parallel, delayed
    import multiprocessing
    available_cpus = multiprocessing.cpu_count()
    if args.n_cores == -1:
        n_cores = available_cpus
    else:
        n_cores = min(args.n_cores, available_cpus)

    # Classify kataegis
    def data_producer():
        for patient, patient_df in mut_df.groupby([PATIENT]):
            for chrom, chrom_df in patient_df.groupby([CHR]):
                imd = chrom_df[IMD].tolist()
                imd[0] = 0
                yield imd, patient, chrom
                
    def sort_patient_chrom(p, chrom):
        if chrom == 'X': return sample_index[p], 23
        elif chrom == 'Y': return sample_index[p], 24
        else: return sample_index[p], int(chrom)
    
    results = Parallel(n_jobs=n_cores, verbose=10)(delayed(find_kataegis_loci)(imd, s, chrom)
                                                   for imd, s, chrom in data_producer())
    results.sort(key=lambda k: sort_patient_chrom(k[1], k[2]))
    mut_df[KATAEGIS] = [ c for sample_chrom_c, _, _ in results for c in sample_chrom_c ]
    
    logger.info('- Identified %s mutations participating in kataegis loci' % mut_df[KATAEGIS].sum())

    # Save updated mutations dataframe to file
    logger.info('[Outputting updated dataframe to file]')
    logger.info('- Writing table to %s' % args.output_file)
    mut_df.to_csv(args.output_file, sep='\t', index=False)

if __name__ == '__main__': run( get_parser().parse_args(sys.argv[1:]) )
