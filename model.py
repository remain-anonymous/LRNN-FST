from sklearn.metrics import roc_auc_score
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
DTYPE = torch.complex64

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=1, seq_size=512,  feature_size=768):
        super(PositionalEncoding, self).__init__()
        # Create a positional encoding layer with a learnable parameter
        
        
        self.pe = nn.Parameter(torch.zeros(seq_size, feature_size))

    def forward(self, x):

        # Add the positional encoding to x
        x = x + self.pe
        return x
    
class ParallelRNN(nn.Module):
    def __init__(self, input_size, hid_rnn_size, seq_expansion= 1):
        super(ParallelRNN, self).__init__()
      
        self.seq_expansion = seq_expansion
        
        self.input_size = input_size*self.seq_expansion
        self.hid_rnn_size = hid_rnn_size
        
        self.B = nn.Parameter(torch.randn(hid_rnn_size, input_size))
        
        # Parameters for eigenvalues
        self.nu = nn.Parameter(torch.randn(hid_rnn_size))
        self.theta = nn.Parameter(torch.randn(hid_rnn_size))
        


    def forward_recurrent(self, x, h0=None):

        if h0 is None:
            h = torch.ones(x.size()[0], self.hid_rnn_size, dtype=DTYPE).to(x.device)
        else:
            h = h0

        # Orvieto et al. 
        lam = torch.exp(-torch.exp(self.nu) + 1j * self.theta)
        
        seq_len = x.size(1)
        outputs = []

        for t in range(seq_len):
            h = torch.einsum('i,bi->bi',  lam , h)  + torch.einsum('ij,bj->bi', self.B, x[:, t, :]) 
            outputs.append(h)
            
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.reshape(outputs.size()[0],-1, outputs.size()[2] ).mean(dim=2)
            
        outputs = torch.cat([outputs.real.to(torch.float32), outputs.imag.to(torch.float32)], dim = 2)

        return outputs

    def forward_summed(self, x, h0=None):

        if h0 is None:
            h0 = torch.ones(x.size()[0], self.hid_rnn_size, dtype=DTYPE).to(x.device)

        # Orvieto et al.
        lam = torch.exp(-torch.exp(self.nu) + 1j * self.theta)
        
        seq_len = x.size(1)
        
        outputs = [(sum([torch.pow(lam, k) * torch.einsum('ij,bj->bi', self.B, x[:, t-k, :]) for k in range(t+1)]) + torch.pow(lam, t+1) * h0) for t in range(seq_len)]
        
        x = torch.stack(outputs, dim=1).real.to(torch.float32)
        
        x = x.reshape(x.size()[0],-1, x.size()[2] ).mean(dim=2)
            
        return x


    def forward(self, x, h0=None):

        if h0 is None:
            h0 = torch.ones(x.size()[0], self.hid_rnn_size, dtype=DTYPE).to(x.device)
        
        # Orvieto et al.
        lam = torch.exp(-torch.exp(self.nu) + 1j * self.theta)

        seq_len = x.size(1)

        Bx = torch.einsum('ij,btj->bti', self.B, x)

        lam_expanded = lam.unsqueeze(1).unsqueeze(2)
        
        # Compute the powers of lam
        time_diffs = torch.abs(torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0))
        
        lam_powers = torch.pow(lam_expanded.to(x.device), time_diffs.to(x.device))
        
        lam_powers = torch.triu(lam_powers)
        
        outputs = torch.einsum('hst,bsh->bth', lam_powers, Bx.to(DTYPE))
            
        # Adjust for the initial hidden state
        outputs += torch.pow(lam.unsqueeze(0).unsqueeze(1), torch.arange(1,seq_len+1).unsqueeze(0).unsqueeze(2).float().to(x.device)) * h0.unsqueeze(1)
        
        #outputs = outputs.reshape(outputs.size()[0],-1, self.seq_expansion, outputs.size()[2] ).mean(dim=2)
        
        outputs = torch.cat([outputs.real.to(torch.float32), outputs.imag.to(torch.float32)], dim = 2)
        
        #outputs = outputs.abs()
        
        return outputs
    
# Define a custom module for transposing a tensor
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)
    
class Triu(nn.Module):
    def __init__(self):
        super(Triu, self).__init__()
    
    def forward(self, x):
        return torch.tril(x)
    
class ParallelRNNBlock(nn.Module):
    def __init__(self, input_dim=768, out_dim=1, seq_size=512, hid_feature_size=512, hid_seq_size=512, transformer_nhead=None, hid_rnn_size=512, blocks=3, dropout=0.2, seq_expansion= 1, FST=True, transformer= False):
        super(ParallelRNNBlock, self).__init__()
    
        print(seq_size, input_dim )
        self.pos_encoder = PositionalEncoding(seq_size=seq_size,  feature_size=input_dim)
       

        if FST == False and transformer==False:
            
            self.mlp_in = nn.Sequential(
            nn.Linear(input_dim, hid_feature_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_feature_size, hid_feature_size),
            )
            
            self.RNN_blocks = nn.ModuleList()
    
            for _ in range(blocks):
                block = [
                    ParallelRNN(hid_feature_size, hid_rnn_size),
                    nn.Linear(2*hid_rnn_size, hid_feature_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid_feature_size, hid_feature_size),
                    ParallelRNN(hid_feature_size, hid_rnn_size),
                    nn.Linear(2*hid_rnn_size, hid_feature_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid_feature_size, hid_feature_size),
                ]
            
            self.RNN_blocks.append(nn.Sequential(*block))
            
        
        elif FST == False and transformer==True:
            
            self.mlp_in = nn.Sequential(
            nn.Linear(input_dim, hid_feature_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_feature_size, hid_feature_size),
            )
            
            
            self.RNN_blocks = nn.ModuleList()
            
            for _ in range(blocks):
                block = [
                    nn.TransformerEncoderLayer(d_model=hid_feature_size, nhead=transformer_nhead),
                    nn.Linear(hid_feature_size, hid_feature_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid_feature_size, hid_feature_size),
                    nn.TransformerEncoderLayer(d_model=hid_feature_size, nhead=transformer_nhead),
                    nn.Linear(hid_feature_size, hid_feature_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid_feature_size, hid_feature_size),
                ]
                self.RNN_blocks.append(nn.Sequential(*block))
                
        elif FST == True and transformer==False:
            
            self.mlp_in = nn.Sequential(
            nn.Linear(input_dim, hid_feature_size),
            Transpose(1, 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(seq_size, hid_seq_size),
            Transpose(1, 2),
            )
            
            self.RNN_blocks = nn.ModuleList()
    
            for _ in range(blocks):
                block = [
                    ParallelRNN(hid_feature_size, hid_rnn_size),
                    nn.Linear(2*hid_rnn_size, hid_feature_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid_feature_size, hid_feature_size),
                    Transpose(1, 2),
                    ParallelRNN(hid_seq_size, hid_rnn_size),
                    nn.Linear(2*hid_rnn_size, hid_seq_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid_seq_size, hid_seq_size),
                    Transpose(1, 2),
                ]
            
            self.RNN_blocks.append(nn.Sequential(*block))
            
        elif FST == True and transformer==True:
            
            self.mlp_in = nn.Sequential(
            nn.Linear(input_dim, hid_feature_size),
            Transpose(1, 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(seq_size, hid_seq_size),
            Transpose(1, 2),
            )
            
            self.RNN_blocks = nn.ModuleList()
            
            for _ in range(blocks):
                block = [
                    nn.TransformerEncoderLayer(d_model=hid_feature_size, nhead=transformer_nhead),
                    nn.Linear(hid_feature_size, hid_feature_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid_feature_size, hid_feature_size),
                    Transpose(1, 2),
                    nn.TransformerEncoderLayer(d_model=hid_seq_size, nhead=transformer_nhead),
                    nn.Linear(hid_seq_size, hid_seq_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid_seq_size, hid_seq_size),
                    Transpose(1, 2),
                ]
                self.RNN_blocks.append(nn.Sequential(*block))
                
        else:
            print("Model type not correctly specified")
                
            
        
        self.mlp_out = nn.Sequential(
            nn.Linear(hid_feature_size, hid_feature_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_feature_size, out_dim),
        )
        
    def forward(self, x):
        x = self.pos_encoder(x)
        h = self.mlp_in(x)
        for block in self.RNN_blocks:
            h = block(h)
        x =  self.mlp_out(h)
        return x, h


def accuracy(predictions, targets, threshold=0.5):
    """
    Compute accuracy given model outputs (predictions) and targets.
    
    Args:
        predictions (torch.Tensor): The output predictions from the model.
        targets (torch.Tensor): The true labels.
        threshold (float, optional): The threshold to convert predictions to binary labels. Defaults to 0.5.

    Returns:
        float: The accuracy of the predictions.
    """
    # Convert predictions to binary (0 or 1) based on threshold
    pred_labels = (predictions >= threshold).float()

    # Compare with targets to find correct predictions
    correct_preds = (pred_labels == targets).float()

    # Calculate accuracy
    accuracy = correct_preds.sum() / len(correct_preds)

    return accuracy.item()  # Return as Python float

def roc_score(predictions, targets):
    with torch.no_grad():
        """
        Compute ROC (Receiver Operating Characteristic) AUC score given model outputs (predictions) and targets.

        Args:
            predictions (torch.Tensor): The output predictions from the model. These should be probabilities, not binary labels.
            targets (torch.Tensor): The true labels.

        Returns:
            float: The ROC AUC score of the predictions.
        """
        # Validate that predictions are in the form of probabilities
        if predictions.min() < 0 or predictions.max() > 1:
            raise ValueError("Predictions should be probabilities ranging from 0 to 1.")

        # Calculate ROC AUC score
        roc_auc = roc_auc_score(targets.detach().cpu().numpy(), predictions.detach().cpu().numpy())

        return roc_auc