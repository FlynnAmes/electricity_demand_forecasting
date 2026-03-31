from torch.utils.data import Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# quick hack for own local issue for now
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SeqExtractionDataSet(Dataset):
    """ custom dataset used to extract sequential data given requests from the dataloader """

    def __init__(self, data_dict, idx_map, SEQ_LENGTH=24*7):
        
        # set data dict which extract the samples from
        self.data_dict = data_dict
         # set idx map, which use to get total number of samples (in len method)
        self.idx_map = idx_map
        # set the sequence length which gives the length of the context window used to predict the target
        self.SEQ_LENGTH = SEQ_LENGTH


    def __getitem__(self, idx):
        """ takes integer idx from the dataloader and returns the features at all time points in 
        context window SEQ_LENGTH (X_seq), along with the target (y_seq) and corresponding client_id.
         
        Note the use of intermediate index map to extract sequences on a per client basis """

        # idx is a numeric integer returned by the dataloader (which it creates using length of all samples)
        # using this integer here to extract a corresponding  client_id and starting index (of time sequence)
        #  from index map (which in turn contains all possible combinations of client ids and starting positions)
        client_id = self.idx_map[idx][0]
        seq_start_idx = self.idx_map[idx][1]

        # extract the corresponding context window and target
        X_seq = self.data_dict[client_id][seq_start_idx: seq_start_idx + self.SEQ_LENGTH, :]
        # for target, just want usage variable only
        y_seq = self.data_dict[client_id][seq_start_idx + self.SEQ_LENGTH, 0]

        # returning id, so can use if want to evaluate performance on clients
        return X_seq, y_seq, client_id


    def __len__(self):
        """ returns the total number of samples in the Dataset. Used by dataloader """
        
        return len(self.idx_map)


class LSTMNoEmbed(nn.Module):

    def __init__(self, n_features, hidden_size=10):

        super().__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)

        # final fully connected layer, taking hidden states and producing output value for next point in time
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
    

    def forward(self, input_seq_batch):

        # just vanilla LSTM with features
        lstm_output, _ = self.lstm(input_seq_batch)

        # and final feedforwards layer. Note need to extract the output from the final timestep in sequence (because)
        # output form LSTM returns hidden state of final LSTM layer at all sequence timesteps
        return self.fc(lstm_output[:, -1, :]).squeeze(-1)


class ConvergenceTracker():
    """ Tracks training and validation losses of model and uses latter for early stopping """

    def __init__(self, patience, delta):
        
        # tracking training and validation losses during the run
        self.training_losses = []
        self.validation_losses = []

        self.patience = patience
        self.delta = delta

        # track the model and epoch where the loss was last decreasing
        self.prev_dec_model = None
        self.prev_dec_epoch = None
        # track the previous loss, every time the check_aely_stop method is called
        self.prev_loss = None
        # counts num iterations since loss last decreased
        self.no_improve_count = 0
        # flag to stop training if patience exceeded
        self.stop_training = False
    
    def check_early_stop(self, val_loss, model_state_dict, epoch):
        
        if self.prev_loss is None or val_loss < self.prev_loss - self.delta:

            # if model loss is still improving (decreasing), reset the counter
            self.no_improve_count = 0

            # and update saved model (the weights and biases from last model where loss had decreased) 
            self.prev_dec_model = model_state_dict
            self.prev_dec_epoch = epoch

            # then set the previous loss to this one, ready for next method call
            self.prev_loss = val_loss
        else:
            # if loss has not improved since last call, increment the counter
            self.no_improve_count += 1

            # set the previous loss to this one
            self.prev_loss = val_loss

            # if counter exceeds patience, signal to end the run
            if self.no_improve_count >= self.patience:
                self.stop_training = True
    
    def plot_losses(self, save_figure=False, show_figure=True, save_path=''):
        
        plt.figure()
        plt.plot(self.validation_losses, label='validation')
        plt.plot(self.training_losses, label='train')
        plt.legend()
        
        if save_figure:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(save_path / 'losses_in_prog.png')
        if show_figure:
            plt.show()
        
        plt.close()
