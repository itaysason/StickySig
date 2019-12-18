#!/usr/bin/env python

# Model names
# MMM_NAME = 'mmm'
# SIGMA_NAME = 'sigma'
# MODEL_NAMES = [MMM_NAME, SIGMA_NAME]

# Kategis constants
KATAEGIS_MIN_MUT = 6
KATAEGIS_IMD = 1000

# Columns in mutation files
PATIENT = 'Patient'
CATEGORY = 'SBS_96'
CATEGORY_IDX = 'Category index'
IMD = 'Distance to Previous Mutation'
NEAREST_MUT_DIST = 'Distance to Nearest Mutation'
REF = 'Reference Sequence'
CHR = 'Chromosome'
PREV_ALLELE_MATCH='Previous Allele Matches'
POS_START = 'Start Position'
KATAEGIS = 'Kataegis (%s+ mutations; IMD<=%s)' % (KATAEGIS_MIN_MUT, KATAEGIS_IMD)

# Strand-specific columns
NO_STRAND = 'No strand'
SAME_STRAND = 'Same strand'
SAME_ALLELE = 'Same allele'
LEADING = 'Leading strand'
LAGGING = 'Lagging strand'
TEMPLATE = 'Template strand'
NON_TEMPLATE = 'Non-template strand'
REP_TX_SAME_DIR = 'Replication/Transcription same direction'
REP_TX_OPP_DIR = 'Replication/Transcription opposite direction'
STRAND_COLUMNS = [ LEADING, LAGGING, TEMPLATE, NON_TEMPLATE,
                   REP_TX_SAME_DIR, REP_TX_OPP_DIR,
                   SAME_ALLELE, SAME_STRAND]
