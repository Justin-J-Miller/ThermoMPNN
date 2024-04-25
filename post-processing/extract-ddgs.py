import pandas as pd
import numpy as np
import argparse
from os import makedirs

def extract_ddgs(args):
	df = pd.read_csv(args.ddgs)
	origAA = str(args.mut[0])
	resi = int(args.mut[1:-1])
	mutAA = args.mut[-1]

	#Pull out indicies from pandas DF that match mutation and selected residue
	indcs = np.where((df['position']==resi) & (df['mutation']==mutAA))[0]

	#Pull out the associated ddGs
	ddgs = df.loc[indcs]['ddG_pred'].to_numpy()

	#Pull out the WT sequence to cross-validate
	selected_wt_seq = df.loc[indcs]['wildtype'].to_numpy()

	#Error catching!
	if len(np.unique(selected_wt_seq)) > 1:
	    print(f'Warning! Your input pdbs had varying WT sequence at position {resi}.')
	    print(f'Found sequences of: {selected_wt_seq}.')
	    exit()
	    
	if np.all(selected_wt_seq == origAA) == False:
	    print(f'Input residue code {origAA}{resi} does not match the residue \
	    	used for input pdbs: {selected_wt_seq}.')
	    exit()
    
	np.save(f'{origAA}{resi}{mutAA}-ddGs.npy', ddgs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddgs', '-f', required=True,
        help='Path to ddG output file.')
    parser.add_argument('--mut', '-m', type=str, required=True,
    	help='OriginalAA_resSeq_mutatedAA (e.g. S44A.')
    parser.add_argument('--outdir','-o', required=False, default='./',
    	help='Where do you want output saved?')

    args = parser.parse_args()
    extract_ddgs(args)