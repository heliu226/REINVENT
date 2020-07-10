#!/usr/bin/env python

import argparse

import torch
import numpy as np
import time
import os
import random
from rdkit import Chem

from model import RNN
from data_structs import Vocabulary, Experience
from scoring_functions import get_scoring_function
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
from vizard_logger import VizardLog

from scoring_functions import activity_model
from logging_functions import sample_smiles, track_agent_loss

# CLI
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--prior_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--sample_size', type=int, default=10000)
parser.add_argument('--n_steps', type=int, default=5000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--scoring_function', type=str, default='activity_model',
                    choices=['activity_model', 'tanimoto', 'no_sulphur'])
parser.add_argument('--clf_file', type=str)
parser.add_argument('--regularization', action='store_true')
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

def train_agent(scoring_function_kwargs=None,
                save_dir=None, 
                learning_rate=0.0005,
                batch_size=64,
                num_processes=0, ## single process
                sigma=60,
                experience_replay=0):
    ## seed all RNGs for reproducible output
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(args.seed)
    
    voc_file = os.path.join(args.input_dir, 'Voc')
    voc = Vocabulary(init_from_file=voc_file)
    
    start_time = time.time()
    
    Prior = RNN(voc)
    Agent = RNN(voc)
    
    logger = VizardLog('data/logs')
    
    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    prior_file = os.path.join(args.prior_dir, 'Prior.ckpt')
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(prior_file))
        Agent.rnn.load_state_dict(torch.load(prior_file))
    else:
        Prior.rnn.load_state_dict(torch.load(prior_file, map_location=lambda \
                                             storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(prior_file, map_location=lambda \
                                             storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=0.0005)

    # Scoring_function
    if args.scoring_function == 'activity_model':
        scoring_function_kwargs = {'clf_path': args.clf_file}
    else:
        scoring_function_kwargs = {}
    
    scoring_function = get_scoring_function(scoring_function=args.scoring_function, 
                                            num_processes=num_processes,
                                            **scoring_function_kwargs)
    
    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)
    
    # Log some network weights that can be dynamically plotted with the Vizard bokeh app
    logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_ih")
    logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_hh")
    logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "init_weight_GRU_embedding")
    logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "init_weight_GRU_layer_2_b_ih")
    logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "init_weight_GRU_layer_2_b_hh")
    
    # Information for the logger
    step_score = [[], []]
    
    print("Model initialized, starting training...")

    sample_every_steps = 50
    sched_file = os.path.join(args.output_dir, 'loss_schedule.csv')
    for step in range(args.n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size, 
                                                       enable_grad=True)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]
        
        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)
        score = scoring_function(smiles)
        
        ## also calculate % predicted active
        if args.scoring_function == 'activity_model':
            mols = [Chem.MolFromSmiles(sm) for sm in smiles]
            fps = [activity_model.fingerprints_from_mol(mol) for mol in mols]
            predictions = [activity_model.clf.predict(fp) for fp in fps]
            mean_active = np.mean(np.asarray(predictions))
        else:
            mean_active = np.NaN
         
        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience)>4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        if args.regularization:
            loss_p = - (1 / agent_likelihood).mean()
            loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((args.n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
              step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(10):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       smiles[i]))
        # Need this for Vizard plotting
        step_score[0].append(step + 1)
        step_score[1].append(np.mean(score))

        # Log some weights
        logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_ih")
        logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_hh")
        logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "weight_GRU_embedding")
        logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "weight_GRU_layer_2_b_ih")
        logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "weight_GRU_layer_2_b_hh")
        logger.log("\n".join([smiles + "\t" + str(round(score, 2)) for smiles, score in zip \
                            (smiles[:12], score[:12])]), "SMILES", dtype="text", overwrite=True)
        logger.log(np.array(step_score), "Scores")
        
        # log and sample SMILES every n steps
        if step % sample_every_steps == 0:
            track_agent_loss(sched_file, step,
                             agent_likelihood.mean(),
                             prior_likelihood.mean(),
                             augmented_likelihood.mean(),
                             score.mean(),
                             mean_active)
            epoch = -1
            sample_smiles(args.output_dir, args.seed, Agent, 
                          args.sample_size, epoch, step)
    
    # If the entire training finishes, we create a new folder where we save this python file
    # as well as some sampled sequences and the contents of the experinence (which are the highest
    # scored sequences seen during training)
    experience.print_memory(os.path.join(args.output_dir, "memory"))
    torch.save(Agent.rnn.state_dict(), os.path.join(args.output_dir, 'Agent.ckpt'))

    seqs, agent_likelihood, entropy = Agent.sample(256)
    prior_likelihood, _ = Prior.likelihood(Variable(seqs))
    prior_likelihood = prior_likelihood.data.cpu().numpy()
    smiles = seq_to_smiles(seqs, voc)
    score = scoring_function(smiles)
    with open(os.path.join(args.output_dir, "sampled"), 'w') as f:
        f.write("SMILES Score PriorLogP\n")
        for smiles, score, prior_likelihood in zip(smiles, score, prior_likelihood):
            f.write("{} {:5.2f} {:6.2f}\n".format(smiles, score, prior_likelihood))

train_agent()
