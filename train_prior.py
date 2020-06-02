#!/usr/bin/env python

import argparse
import os
import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import decrease_learning_rate
rdBase.DisableLog('rdApp.error')

from logging_functions import track_loss, sample_smiles
from early_stopping import EarlyStopping

# CLI
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--sample_size', type=int, default=10000)
parser.add_argument('--patience', type=int, default=10000)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

def pretrain(restore_from=None):
    """Trains the Prior RNN"""
    ## seed all RNGs for reproducible output
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(args.seed)
    
    # Read vocabulary from a file
    voc_file = os.path.join(args.input_dir, 'Voc')
    voc = Vocabulary(init_from_file=voc_file)

    # Create a Dataset from a SMILES file
    mol_file = os.path.join(args.input_dir, 'mols_filtered.smi')
    moldata = MolData(mol_file, voc)
    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = RNN(voc)

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    # set up early stopping
    early_stop = EarlyStopping(patience=args.patience)

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)
    counter = 0
    log_every_steps = 50
    sample_every_steps = 500
    sched_file = os.path.join(args.output_dir, 'loss_schedule.csv')
    for epoch in range(1, 6):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):
            # increment counter
            counter += 1

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]))
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
            
            # log and sample SMILES every n steps
            if counter % log_every_steps == 0:
                track_loss(sched_file, Prior, moldata, epoch, 
                           counter, loss.item(), 128)
            if counter % sample_every_steps == 0:
                sample_smiles(args.output_dir, args.sample_idx, Prior, 
                              args.sample_size, epoch, counter)

            # check early stopping
            validation = moldata.get_validation(128).long()
            validation_logp, _ = Prior.likelihood(validation)
            validation_loss = validation_logp.mean().detach()
            model_filename = "Prior.ckpt"
            model_file = os.path.join(args.output_dir, model_filename)
            early_stop(validation_loss.item(), Prior, model_file, counter)
        
            if early_stop.stop:
                break

        # log and sample SMILES every epoch
        track_loss(sched_file, Prior, moldata, epoch,
                   counter, loss.item(), 128)
        sample_smiles(args.output_dir, args.sample_idx, Prior, 
                      args.sample_size, epoch, counter)

    # append information about final training step
    sched = pd.DataFrame({'epoch': [None],
                          'step': [early_stop.step_at_best],
                          'outcome': ['training loss'], 
                          'value': [early_stop.best_loss]})
    sched.to_csv(sched_file, index=False, mode='a', header=False)

pretrain()