
from ReportPretraining import QuickLoader
import torch
import torch.nn as nn
# imports
import snntorch as snn
from snntorch import surrogate

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
# imports
import snntorch as snn
from snntorch import surrogate
from torchmetrics.classification import F1Score, Accuracy, ConfusionMatrix

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SpikeLayer(nn.Module):
    def __init__(self, input_size,hidden_size, output_size,spike_grad, beta, kernel_size = 1):
        assert hidden_size % kernel_size == 0, '{} % {} != 0'.format(hidden_size,kernel_size)
        assert isinstance(kernel_size, int)
        super().__init__()
        self.kernel_size = kernel_size
        self.linear = nn.Linear(input_size, hidden_size) # takes an input of 3197 and outputs 128
        self.lif = snn.Leaky(beta=torch.rand(hidden_size),
                               threshold = 1.0,
                               learn_beta=True,
                               learn_threshold = True, 
                               learn_graded_spikes_factor = True,
                               spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=torch.rand(output_size),
                               threshold = 1.0,
                               learn_beta=True,
                               learn_threshold = True, 
                               learn_graded_spikes_factor = True,
                               spike_grad=spike_grad)
        #self.lif = snn.Synaptic(alpha=alpha, beta=beta)
        self.out = nn.Linear(hidden_size//kernel_size, output_size)
        self.dropout = nn.Dropout(0.0)
        
    def forward(self, x):
        mem1 = self.lif.init_leaky()
        mem2 = self.lif2.init_leaky()
        cur1 = F.max_pool1d(self.dropout(self.linear(x)), self.kernel_size)
        spk1, mem1 = self.lif(cur1, mem1) 
        cur2 = self.dropout(self.out(spk1))
        #print(mem1.shape)
        spk2, mem2 = self.lif2(cur2, mem2) 

        return spk2.view(x.size(0),-1)
   

class Eye(nn.Module):
    def __init__(self, input_size, 
                 hidden_size, 
                 output_size,
                spike_grad,
                beta):
        super().__init__()
        self.spike = SpikeLayer(input_size = input_size,
                    hidden_size = hidden_size,
                    output_size = output_size,
                    spike_grad=spike_grad,
                     beta =beta)
        self.ln = nn.LayerNorm(input_size)
    def forward(self, x):
        return self.spike(self.ln(x))


class ANGEL(nn.Module):
    def __init__(self, num_bands = 2, 
                    input_size = 200,
                    hidden_size = 5000,
                    output_size = 200, 
                    beta = None,
                    spike_grad = None,
                    num_classes = 22):
        super().__init__()

        self.bands = nn.ModuleList([Eye(input_size,
                                        hidden_size,
                                        output_size,
                                        spike_grad = spike_grad,
                                         beta=torch.rand(hidden_size), 
                                         ) for _ in range(2)])
        self.ln_out = nn.LayerNorm(output_size*num_bands)
        self.classifier = SpikeLayer(output_size*num_bands, 
                                     hidden_size= hidden_size,
                                     spike_grad=spike_grad,
                                     beta = torch.rand(hidden_size),
                                     output_size=num_classes,
                                     )
        self.classifier = nn.Linear(output_size*num_bands,num_classes)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self,x):
        outs = []
        for i in range(x.size(-1)):
            outs.append(self.bands[i](x[:,:,i]))
        outs = torch.concat(outs, axis = -1)
        x = outs
        x = self.ln_out(x)
        x = self.classifier(x)
        return x
    # shape of outs should be [(bsz,spikeout),[bsz, spikeout]]

class test_transforms:
    
    def __call__(self, sample):
        x = sample['data']
        if torch.rand(1) <=0.5:
            x = x + np.random.normal(0,x.std(), x.shape)
         
        if torch.rand(1) <= 0.5:
            x = x + torch.normal(0,1, size = (1,)).item() 
        if torch.rand(1) <= 0.5:
            x[np.random.randint(10,199):200,0] = 0
            x[np.random.randint(10,199):200,1] = 0
        if torch.rand(1) <= 0.5:
            x = torch.roll(x,shifts= torch.randint(0,200, size = (1,)).item(), dims = 0)
            
       # if torch.rand(1) <= 0.5:
      #      x = torch.roll(x,shifts= torch.randint(0,1, size = (1,)).item(), dims = 1)
            
        sample['data'] =  x
        return sample
    
from scipy import interpolate

class resample:
    def __call__(self, sample):
        for i in range(2):
            data = sample['data'][:,i]
            time = sample['time'][:,i] 
            f = interpolate.interp1d(time, data, kind='nearest', 
                                    bounds_error=False, fill_value=0)
            t_min = min(time)
            t_max = max(time)
            time_step = t_max /(data.size(0)) # Regular interval (e.g., every 0.5 time units)
            time_regular = np.arange(t_min, t_max, time_step)
            data_regular = f(time_regular)
            sample['data'][:,i]  = torch.tensor(data_regular)[:200]
            sample['time'][:,i] = torch.tensor(time_regular)[:200]
        return sample
        
tr = [#resample(),
      test_transforms()
      ]
ql = QuickLoader(batch_size=512, transforms = tr, train_apply_transform=True)
train_dataloader = ql.train
validation_dataloader = ql.validation
def train_model():
    # Set device
    device = torch.device("cuda:2" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    
    # Initialize model and training components
    model = ANGEL(2, spike_grad=None)
    model.to(device)
    from src.losses.FocalLoss import FocalLoss
    #criterion =FocalLoss(gamma = 10,alpha = [100/22], task_type='multi-class',num_classes=22) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Import torchmetrics
    
    # Initialize metrics
    f1_metric = F1Score(task="multiclass", num_classes=22, average="macro").to(device)
    accuracy_metric = Accuracy(task="multiclass", num_classes=22).to(device)
    
    # Early stopping parameters
    num_epochs = 1000
    patience = 100
    delta = 0.0001
    best_f1 = 0
    counter = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        
        for data in train_dataloader:
            inputs = data['data'].float().to(device)
            labels = data['labels'].long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            
            # Reset metrics
            f1_metric.reset()
            accuracy_metric.reset()
            
            for data in validation_dataloader:
                inputs = data['data'].float().to(device)
                labels = data['labels'].long().to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                
                # Update metrics
                f1_metric(predictions, labels)
                accuracy_metric(predictions, labels)
            
            # Compute final metrics
            avg_val_loss = total_val_loss / len(validation_dataloader)
            current_f1 = f1_metric.compute().item()
            current_accuracy = accuracy_metric.compute().item()
            
            # Early stopping logic
            if current_f1 > best_f1 + delta:
                best_f1 = current_f1
                counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} F1: {best_f1:.4f} (Best) Acc: {current_accuracy:.4f}", end='\r')
            else:
                counter += 1
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} F1: {best_f1:.4f} (Best) Acc: {current_accuracy:.4f}", end='\r')
                if counter >= patience:
                    early_stop = True
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        if early_stop:
            break
    
    # Evaluation on validation set using best model
    print(f"\nTraining completed. Loading best model with F1: {best_f1:.4f}")


def test_model():
    device = 'cuda:2'
    model = ANGEL(2, spike_grad=None)
    model.to(device)
    model.to('cpu')
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Initialize confusion matrix for final evaluation
    from torchmetrics.classification import MulticlassConfusionMatrix
    confusion = MulticlassConfusionMatrix(num_classes=22)
    f1_metric = F1Score(task="multiclass", num_classes=22, average='macro')
    
    with torch.no_grad():
        all_preds = []
        all_labels = []
        
        for data in tqdm(validation_dataloader):
            inputs = data['data'].float().to('cpu')
            labels = data['labels'].long().to('cpu')
            
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            
            all_preds.append(predictions)
            all_labels.append(labels)
        
        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        from src.utils.data.AlerceDictionaries import ZTF_TAXONOMY
        print(classification_report(all_labels, all_preds, digits = 4, target_names=ZTF_TAXONOMY().keys()))
        
        all_preds = []
        all_labels = []
        for data in tqdm(ql.test):
                inputs = data['data'].float().to('cpu')
                labels = data['labels'].long().to('cpu')
                
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                
                all_preds.append(predictions)
                all_labels.append(labels)
            
        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        from src.utils.data.AlerceDictionaries import ZTF_TAXONOMY
        print(classification_report(all_labels, all_preds, digits = 4, target_names=ZTF_TAXONOMY().keys()))

train_model()

#test_model()