import torch

class EarlyStopping():
    """
    Monitor the training process to stop training early if the model shows
    evidence of beginning to overfit the validation dataset, and save the
    best model. 
    
    Note that patience here is measured in steps, rather than in epochs,
    because the size of an epoch will not be consistent if the size of the 
    dataset changes.
    
    Inspired by:
    https://github.com/Bjarten/early-stopping-pytorch
    https://github.com/fastai/fastai/blob/master/courses/dl2/imdb_scripts/finetune_lm.py
    """
    
    def __init__(self, patience=100):
        """
        Args:
            model: the PyTorch model being trained 
            output_file: (str) location to save the trained model
            patience: (int) if the validation loss fails to improve for this
              number of consecutive batches, training will be stopped 
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.step_at_best = 0
        self.stop = False
        print("instantiated early stopping with patience=" + \
              str(self.patience))
    
    def __call__(self, val_loss, model, output_file, step_idx):
        # do nothing if early stopping is disabled
        if self.patience > 0:
            if self.best_loss is None:
                self.best_loss = val_loss
                self.step_at_best = step_idx
                self.save_model(model, output_file)
            elif val_loss >= self.best_loss:
                # loss is not decreasing
                self.counter += 1
                if self.counter >= self.patience:
                    self.stop = True
                    print("stopping early with best loss " + \
                          str(self.best_loss))
            else: 
                # loss is decreasing
                self.best_loss = val_loss
                self.step_at_best = step_idx
                ## reset counter
                self.counter = 0
                self.save_model(model, output_file)
    
    def save_model(self, model, output_file):
        torch.save(model.rnn.state_dict(), output_file)
