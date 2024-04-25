import mdtraj as md
import pandas as pd
from glob import glob
from pathlib import Path
import argparse
from os import makedirs

def preprocess(args):
	makedirs(args.outdir, exist_ok=True)
	#Convert xtc to pdbs
	trj = md.load(args.xtc,top=args.top)
	for i, pdb in enumerate(trj):
	    pdb.save_pdb(f'{args.outdir}/state{str(i).zfill(5)}.pdb')

	#Prepare csv
	PDB_paths = sorted(glob(f'{args.outdir}/state*.pdb'))
	PDBs = [Path(pdb).stem for pdb in PDB_paths[:3]]
	MUT = ['A']*len(PDBs) #This doesn't matter and will be overwritten in SSM.py
	SEQ = trj.top.to_fasta()*len(PDBs)
	df = pd.DataFrame({'PDB':PDBs, 'MUT':MUT, 'SEQ':SEQ})
	df.to_csv(f'{args.outdir}/centers.csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xtc', '-f', required=True,
        help='Path to centers xtc file.')
    parser.add_argument('--top', '-s', required=True,
    	help='Path to topology file to load centers.')
    parser.add_argument('--outdir','-o', required=False, default='./',
    	help='Where do you want output saved?')

    args = parser.parse_args()
    preprocess(args)