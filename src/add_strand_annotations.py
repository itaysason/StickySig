#!/usr/bin/env python

# Load required modules
import sys, os, argparse, json, pandas as pd, logging, numpy as np

# Load our modules
from data_utils import get_logger
from constants import *

# Command-line argument parser
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--mutations_file', type=str, required=True)
    parser.add_argument('-sf', '--strand_info_file', type=str, required=True)
    parser.add_argument('-of', '--output_file', type=str, required=True)
    parser.add_argument('-v', '--verbosity', type=int, required=False,
                        default=logging.INFO)
    return parser

# Main
def run( args ):
    # Set up logger
    logger = get_logger(args.verbosity)
    logger.info('[Loading input data]')

    # Load strand information and process into dictionaries
    strand_df = pd.read_csv(args.strand_info_file, sep='\t')
    strand_keys = list(zip(strand_df['tChr'], strand_df['tPos0']))
    is_left = dict(zip(strand_keys, map(bool, strand_df['tIsLeft'])))
    is_right = dict(zip(strand_keys, map(bool, strand_df['tIsRight'])))
    is_tx_plus = dict(zip(strand_keys, map(bool, strand_df['tTxPlus'])))
    is_tx_minus = dict(zip(strand_keys, map(bool, strand_df['tTxMinus'])))

    logger.info('- Loaded strand information in %s bins' % len(strand_df))

    # Define the main categorization by strand function. We return six categories:
    # 1) Lagging strand
    # 2) Leading strand
    # 3) Template (genic regions only)
    # 4) Non template
    # 5) Transcription and replication in the same direction (genic regions only)
    # 6) Transcription and replication in opposite directions
    def categorize_by_stand(mut):
        # Get the chromosome and bin start positions
        chrom = 'chr%s' % mut[CHR]
        pos = mut[POS_START]
        bin_start = np.floor(pos/20000)*20000
        
        strand_key = sk = (chrom, bin_start)
        
        is_ref_plus = mut[REF] in {'C', 'T'}
        is_ref_minus = not is_ref_plus
        
        # Classify leading/lagging
        if not(is_left[sk] or is_right[sk]):
            leading = np.nan
            lagging = np.nan
        elif (is_ref_plus and is_left[sk]) or (is_ref_minus and is_right[sk]):
            lagging = 0
            leading = 1
        else:
            lagging = 1
            leading = 0

        # Classify template/non-template
        if not(is_tx_plus[sk] or is_tx_minus[sk]):
            template = np.nan
            non_template = np.nan
        elif (is_ref_plus and is_tx_plus[sk]) or (is_ref_minus and is_tx_minus[sk]):
            template = 0
            non_template = 1
        else:
            template = 1
            non_template = 0

        # Classify tx/rep as same/opposite
        if not(is_left[sk] or is_right[sk]) or not(is_tx_plus[sk] or is_tx_minus[sk]):
            rep_tx_same = np.nan
            rep_tx_opposite = np.nan
        elif (is_right[sk] and is_tx_plus[sk])  or (is_left[sk] and is_tx_minus[sk]):
            rep_tx_same = 1
            rep_tx_opposite = 0
        else:
            rep_tx_same = 0
            rep_tx_opposite = 1

        return leading, lagging, template, non_template, rep_tx_same, rep_tx_opposite

    # Load mutations
    mut_df = pd.read_csv(args.mutations_file, sep='\t')
    columns = mut_df.columns
    samples = set(mut_df[PATIENT])
    logger.info('- Loaded %s mutations from %s samples' % (len(mut_df), len(samples)))

    # Add strand information
    logger.info('[Adding strand information]')
    strand_categories = list(zip(*[ categorize_by_stand(m) for _, m in mut_df.iterrows() ]))
    strand_category_names = [LEADING, LAGGING, TEMPLATE, NON_TEMPLATE, REP_TX_SAME_DIR, REP_TX_OPP_DIR]
    for i, strand_cat in enumerate(strand_category_names):
        mut_df[strand_cat] = strand_categories[i]
        logger.info('- %s mutations "%s"' % (np.nansum(strand_categories[i]), strand_cat))

    # Save updated mutations dataframe to file
    logger.info('[Outputting updated dataframe to file]')
    logger.info('- Writing table to %s' % args.output_file)
    mut_df = mut_df[columns.tolist() + strand_category_names]
    mut_df.to_csv(args.output_file, sep='\t', index=False)

if __name__ == '__main__': run( get_parser().parse_args(sys.argv[1:]) )

