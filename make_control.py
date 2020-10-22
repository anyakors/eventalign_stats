import numpy as np
import pandas as pd

from itertools import product
from utils import *

df_control = new_control_5("/Volumes/anyaext/NA12878/bc_guppy/batch_1/eventalign_collapsed.tsv/out_eventalign_collapse.tsv",'NA12878')

print(df_control.head())

b = ['A','T','C','G']
kmers = [''.join(x) for x in list(product(b, repeat=5))]

for kmer in kmers:
    indx = df_control['kmer'] == kmer
    df_control_indx = df_control[indx]
    if len(df_control_indx)==0:
        print('kmer {} was not found'.format(kmer))

df_control.to_csv('./tables/control_5kmer_rna.csv')