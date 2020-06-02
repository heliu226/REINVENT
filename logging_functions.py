import os
import pandas as pd

def track_loss(output_file, model, dataset, epoch, step_idx,
               training_loss, batch_size):
    """
    Log model training and validation losses to a file. 
    
    Args:
        output_file: the file to write the training schedule to
        model: the model currently being trained
        dataset: the dataset being trained on
        epoch: current epoch of model training
        step_idx: current step (minibatch index) overall
        training_loss: training loss at the last step
        batch_size: size of each minibatch; used to calculate validation loss 
            and sample SMILES
    """
    
    sched = pd.DataFrame()
    validation, lengths = dataset.get_validation(batch_size)
    validation_loss = model.loss(validation, lengths).mean().detach().item()
    sched = pd.DataFrame({'epoch': epoch + 1, 'step': step_idx,
                          'outcome': ['training loss', 'validation loss'], 
                          'value': [training_loss, validation_loss]})
    
    # write training schedule (write header if file does not exist)
    if not os.path.isfile(output_file) or step_idx == 0:
        sched.to_csv(output_file, index=False)
    else:
        sched.to_csv(output_file, index=False, mode='a', header=False)

def sample_smiles(output_dir, sample_idx, model, sample_size, epoch, 
                  step_idx): 
    """
    Sample a set of SMILES from a trained model, and write them to a file
    
    Args:
        output_dir: directory to write output files to
        sample_idx: index of the SMILES sample being trained on; included in
            all output fles
        model: the model currently being trained
        sample_size: the number of SMILES to sample and write 
        epoch: current epoch of model training
        step_idx: current step (minibatch index) overall, or 'NA' at the end
            of an epoch
    """
    
    sampled_smiles = []
    while len(sampled_smiles) < sample_size:
        sampled_smiles.extend(model.sample(100, return_smiles=True))
    
    # set up output filename
    if step_idx == "NA": 
        # writing by epoch: don't include batch index
        smiles_filename = "sample-" + str(sample_idx + 1) + \
            "-epoch=" + str(epoch + 1) + "-SMILES.smi"
    else: 
        # writing by step: calculate overall step
        smiles_filename = "sample-" + str(sample_idx + 1) + \
            "-epoch=" + str(epoch + 1) + "-step=" + str(step_idx) + \
            "-SMILES.smi"
    
    # write to file
    smiles_file = os.path.join(output_dir, smiles_filename)
    write_smiles(sampled_smiles, smiles_file)    

def write_smiles(smiles, smiles_file):
    """
    Write a list of SMILES to a line-delimited file.
    """
    # write sampled SMILES
    with open(smiles_file, 'w') as f:
        for sm in smiles:
            _ = f.write(sm + '\n')