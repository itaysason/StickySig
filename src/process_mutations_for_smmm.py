#!/usr/bin/env python

# Load required modules
import sys, os, argparse, json, pandas as pd, logging, numpy as np
from collections import defaultdict
from itertools import permutations

# Load our modules
from data_utils import get_logger
from constants import *

# Command-line argument parser
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--mutations_file', type=str, required=True)
    parser.add_argument('-sf', '--signatures_file', type=str, required=True)
    parser.add_argument('-of', '--output_file', type=str, required=True)
    parser.add_argument('-v', '--verbosity', type=int, required=False,
                        default=logging.INFO)
    return parser

# Main
def run( args ):
    # Set up logger
    logger = get_logger(args.verbosity)
    logger.info('[Loading input data]')

    # Load the signatures
    sig_df = pd.read_csv(args.signatures_file, sep='\t', index_col=0)
    categories = list(sig_df.columns)
    category_index = dict(zip(categories, range(len(categories))))

    logger.info('- Loaded %s x %s signature matrix' % sig_df.values.shape)

    # Load the mutations
    use_strand_cols = list(set(STRAND_COLUMNS)-{SAME_ALLELE, SAME_STRAND})
    mut_df = pd.read_csv(args.mutations_file, sep='\t',
                         usecols=[PATIENT, CATEGORY, REF, CHR, KATAEGIS] + use_strand_cols)
    samples = sorted(set(mut_df[PATIENT]))

    logger.info('- Loaded %s mutations in %s samples' % (len(mut_df), len(samples)))

    # Add the category index and create sequences of mutations
    logger.info('[Processing data into sMMM format]')
    mut_df[CATEGORY_IDX] = mut_df.apply(lambda r: category_index[r[CATEGORY]],
                                    axis='columns')
    mut_df[SAME_ALLELE] = mut_df[REF]
    #mut_df[SAME_STRAND] = mut_df[REF].replace({REF: {'A': 0, 'C': 1, 'G': 0, 'T': 1}})
    mut_df[SAME_STRAND] = mut_df[REF]
    mut_df = mut_df.replace({SAME_STRAND: {'A': 0, 'C': 1, 'G': 0, 'T': 1}})

    sampleToStrandMatch = { s: {} for s in samples }
    sampleToKataegis = {}
    for s, sample_df in mut_df.groupby([PATIENT]):
        # Go sample by sample to figure out when adjacent mutations match
        sampleToKataegis[s] = sample_df[KATAEGIS].tolist()
        for col in STRAND_COLUMNS:
            # Compare adjacent mutations
            xs = np.array(sample_df[col].tolist())
            if col in {SAME_ALLELE, SAME_STRAND}:
                matched = xs == np.roll(xs, 1)
            else:
                # Convert NaNs to -1 because otherwise they could
                # break the comparison
                xs[np.isnan(xs)] = 0
                xs = xs.astype(bool)
                matched = xs & np.roll(xs, 1)

            # Finally, set the first position to False (to be safe)
            matched[0] = False
            
            sampleToStrandMatch[s][col] = matched.astype(int).tolist()
        
        sampleToStrandMatch[s][NO_STRAND] = [0] + [1] * (len(sample_df)-1)

    sampleToSequence = dict( (s, list(map(int, s_df[CATEGORY_IDX])))
                                 for s, s_df in mut_df.groupby([PATIENT]) )
    chromosomeMap = dict( (s, list(s_df[CHR]))
                          for s, s_df in mut_df.groupby([PATIENT]) )

    # Save to JSON
    logger.info('- Saving to file %s' % args.output_file)
    with open(args.output_file, 'w') as OUT:
        output = dict(sampleToSequence=sampleToSequence,
                      sampleToStrandMatch=sampleToStrandMatch,
                      sampleToKataegis=sampleToKataegis,
                      chromosomeMap=chromosomeMap,
                      samples=samples, categories=categories,
                      params=vars(args))
        json.dump( output, OUT )

if __name__ == '__main__': run( get_parser().parse_args(sys.argv[1:]) )
