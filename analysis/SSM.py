import pandas as pd
from tqdm import tqdm
import torch
from omegaconf import OmegaConf

import os
import sys
sys.path.append('../')
from datasets import MegaScaleDataset, ddgBenchDataset, FireProtDataset, Mutation, CustomDataset
from protein_mpnn_utils import loss_smoothed, tied_featurize
from train_thermompnn import TransferModelPL
from model_utils import featurize
from thermompnn_benchmarking import compute_centrality, ProteinMPNNBaseline, get_trained_model, ALPHABET


def get_ssm_mutations(pdb):
        # make mutation list for SSM run
    mutation_list = []
    for seq_pos in range(len(pdb['seq'])):
        wtAA = pdb['seq'][seq_pos]
        # check for missing residues
        if wtAA != '-':
            # add each mutation option
            for mutAA in ALPHABET[:-1]:
                mutation_list.append(wtAA + str(seq_pos) + mutAA)
        else:
            mutation_list.append(None)

    return mutation_list


def retrieve_best_mutants(df_slice, allow_cys=True):
    # check for best mutant at each position
    pos_list = df_slice.position.unique()
    best_res_list = []
    for p in pos_list:
        p_slice = df_slice.loc[df_slice['position'] == p].reset_index(drop=True)
        if not allow_cys:  # filter out cysteine option
            p_slice = p_slice.loc[p_slice['mutation'] != 'C'].reset_index(drop=True)
        min_row = p_slice.iloc[pd.to_numeric(p_slice['ddG_pred']).idxmin()]
        best_res_list.append(min_row['mutation'])
    return best_res_list


def main(cfg, args):
    """Inference script that does site-saturation mutagenesis for a given protein"""
    # define config for model loading
    os.makedirs(args.outdir, exist_ok=True)

    config = {
        'training': {
            'num_workers': 8,
            'learn_rate': 0.001,
            'epochs': 100,
            'lr_schedule': True,
        },
        'model': {
            'hidden_dims': [64, 32],
            'subtract_mut': True,
            'num_final_layers': 2,
            'freeze_weights': True,
            'load_pretrained': True,
            'lightattn': True,
            'lr_schedule': True,
        }
    }

    cfg = OmegaConf.merge(config, cfg)

    models = {
        'ProteinMPNN': ProteinMPNNBaseline(cfg, version='v_48_020.pt'),
        "ThermoMPNN": get_trained_model(model_name='thermoMPNN_default.pt',
                                        config=cfg)
    }

    datasets = {
        args.prot_name: CustomDataset(cfg, pdb_dir=args.inp_dir, csv_fname=f'{args.inp_dir}/centers.csv')
    }

    max_batches = None
    row = 0

    for name, model in models.items():
        model = model.eval()
        model = model.cuda()
        for dataset_name, dataset in datasets.items():
            raw_pred_df = pd.DataFrame(columns=['wildtype', 'mutation','ddG_pred', 'position'])

            print('Running model %s on dataset %s' % (name, dataset_name))
            for i, batch in enumerate(tqdm(dataset)):
                mut_pdb, mutations = batch

                #Pull the first residue number out so we can renumber the output according to orginal pdb
                first_resi_n = int(mut_pdb[0]['resn_list'][0])

                # generate all SSM mutations
                mutation_list = get_ssm_mutations(mut_pdb[0])
                final_mutation_list = []

                # build into list of Mutation objects
                for n, m in enumerate(mutation_list):
                  if m is None:
                    final_mutation_list.append(None)
                    continue
                  m = m.strip()  # clear whitespace
                  wtAA, position, mutAA = str(m[0]), int(str(m[1:-1])), str(m[-1])
                  assert wtAA in ALPHABET, f"Wild type residue {wtAA} invalid, please try again with one of the following options: {ALPHABET}"
                  assert mutAA in ALPHABET, f"Wild type residue {mutAA} invalid, please try again with one of the following options: {ALPHABET}"
                  mutation_obj = Mutation(position=position, wildtype=wtAA, mutation=mutAA, 
                                          ddG=None, pdb=mut_pdb[0]['name'])
                  final_mutation_list.append(mutation_obj)

                pred, _ = model(mut_pdb, final_mutation_list)

                # calculation of N neighbors
                if args.centrality:
                    coord_chain = [c for c in mut_pdb[0].keys() if 'coords' in c][0]
                    chain = coord_chain[-1]
                    neighbors = compute_centrality(mut_pdb[0][coord_chain], basis_atom='CA', backup_atom='C', chain=chain, radius=10.)

                for mut, out in zip(final_mutation_list, pred):
                    if mut is not None:
                        col_list = ['ddG_pred', 'position', 'wildtype', 'mutation', 'pdb']
                        val_list = [out["ddG"].cpu().item(), mut.position+first_resi_n, mut.wildtype,
                                    mut.mutation, mut.pdb.strip('.pdb')]
                        for col, val in zip(col_list, val_list):
                            raw_pred_df.loc[row, col] = val


                        if args.verbose == True:
                            raw_pred_df['WT Seq']=""
                            raw_pred_df['Model']=""
                            raw_pred_df['Dataset']=""
                            raw_pred_df['best_AA']=""

                            if args.centrality:
                                raw_pred_df.loc[row, 'neighbors'] = neighbors[mut.position].cpu().item()
                                raw_pred_df['neighbors']=""
                            raw_pred_df.loc[row, 'Model'] = name
                            raw_pred_df.loc[row, 'Dataset'] = dataset_name
                            if 'Megascale' not in dataset_name:
                                key = mut.pdb
                            else:
                                key = mut.pdb + '.pdb'
                            raw_pred_df.loc[row, 'WT Seq'] = dataset.wt_seqs[key]
                        row += 1

                if args.pick_best:
                    # retrieve BEST mutation at each spot and save to DF
                    current_slice = raw_pred_df.loc[raw_pred_df['pdb'] == mut.pdb.strip('.pdb')]
                    best_res_list = retrieve_best_mutants(current_slice, allow_cys=args.include_cys)
                    for p, b in zip(current_slice.position.unique(), best_res_list):
                        raw_pred_df.loc[(raw_pred_df['pdb'] == mut.pdb.strip('.pdb')) & (raw_pred_df['position'] == p), 'best_AA'] = b

                    # only keep one row per position for AA pattern detection
                    raw_pred_df['dupe_detector'] = raw_pred_df['pdb'] + raw_pred_df['position'].astype(str)
                    raw_pred_df = raw_pred_df.drop_duplicates(subset=['dupe_detector'], keep='first')

                else:
                    if not args.include_cys:
                        raw_pred_df = raw_pred_df.loc[raw_pred_df['mutation'] != 'C']

                if max_batches is not None and i >= max_batches:
                    break

                print('Completed protein:', mut.pdb)
                print('Mutations processed:', raw_pred_df.shape)

            raw_pred_df = raw_pred_df.reset_index(drop=True)
            raw_pred_df.to_csv(f'{args.outdir}/{name}_{dataset_name}_SSM_preds.csv', index=False)
            del raw_pred_df


if __name__ == "__main__":
    cfg = OmegaConf.load("../local.yaml")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_dir','-f', required=True,
        help='Path to output from preprocess-SSM-xtcs.py.')
    parser.add_argument('--verbose', default=False, type=bool,
        help='Do you want verbose output?')
    parser.add_argument('--prot_name', default='MSM', type=str,
        help='Name to prepend output file with.')
    parser.add_argument('--outdir', '-o', default='./',
        help='Where do you want output saved?')
    parser.add_argument('--pick_best', action='store_true', default=False,
                        help='Keep only the BEST mutation at each position')
    parser.add_argument('--include_cys', action='store_true', default=False,
                        help='Include cysteine as potential mutation option.'
                             'Due to assay artifacts, mutations to cys are predicted poorly.')
    parser.add_argument('--centrality', action='store_true', default=False,
                        help='Calculate centrality value for each residue (# neighbors). '
                             'Only used if --keep_preds is enabled.')
    args = parser.parse_args()

    with torch.no_grad():
        main(cfg, args)
