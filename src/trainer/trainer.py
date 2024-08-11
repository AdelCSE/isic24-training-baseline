import gc
import os
import sys
import copy
import time
import torch
from tqdm import tqdm
from torch import nn
from typing import Any, Optional

sys.path.append('../..')
from src.utils import compute_metrics

class Trainer:

    def __init__(self,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 criterion: Optional[torch.nn.Module]= None,
                 scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None,
                 device: str = 'cuda',
                 weights_folder: Optional[str] = None
                 ):
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.model = None
        self.history = self.init_history()
        self.weights_folder = weights_folder

    
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        return self
    
    def set_criterion(self, criterion: torch.nn.Module):
        self.criterion = criterion
        return self
    
    def set_scheduler(self, scheduler: torch.optim.lr_scheduler.CosineAnnealingLR):
        self.scheduler = scheduler
        return self
    
    def set_device(self, device: str):
        self.device = device
        return self
    
    def init_history(self) -> dict:
        history = {
            'train': {},
            'val': {}
        }

        history['train']['loss'] = []
        history['train']['acc'] = []
        history['train']['f1'] = []
        history['train']['auc'] = []
        history['train']['pauc'] = []
        history['train']['lr'] = []
        history['train']['epoch'] = []

        history['val']['loss'] = []
        history['val']['acc'] = []
        history['val']['f1'] = []
        history['val']['auc'] = []
        history['val']['pauc'] = []
        history['val']['epoch'] = []

        return history
    
    def reset(self) -> None:
        self.history = self.init_history()

    def append_to_history(self, results: dict[str, torch.Tensor], to: str) -> None:
        for key, value in results.items():
            self.history[to][key].append(value)

    
    def train_one_epoch(self, model, optimizer, criterion, scheduler, loader, device, n_accum, epoch):
        
        model.train()

        y_true = []
        y_prob = []

        running_loss = 0.0
        dataset_size = 0.0

        t = tqdm(enumerate(loader), total=len(loader))
        for idx, data in t:
            images = data['image'].to(device, dtype=torch.float)
            targets = data['target'].to(device, dtype=torch.float)

            batch_size = images.shape[0]

            # Forward pass
            outputs = model(images).squeeze()

            # Compute the loss
            loss = criterion(outputs, targets) / n_accum

            # Backward pass
            loss.backward()

            if (idx + 1) % n_accum == 0:
                # Update the weights
                optimizer.step()

                # Zero the gradients
                optimizer.zero_grad()

                # Update the scheduler
                if scheduler is not None:
                    scheduler.step()

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            
            y_true.extend(targets.cpu().numpy())
            y_prob.extend(outputs.detach().cpu().numpy())
        
        epoch_loss = running_loss / dataset_size
        acc, f1, auc, pauc = compute_metrics(y_true, y_prob)

        print(f'Train Epoch: {epoch}, Loss: {epoch_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, PAUC: {pauc:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        gc.collect()

        return epoch_loss, acc, f1, auc, pauc

    @torch.inference_mode()
    def valid_one_epoch(self, model, criterion, loader, device, epoch):
        
        model.eval()

        y_true = []
        y_prob = []

        running_loss = 0.0
        dataset_size = 0.0

        t = tqdm(loader, total=len(loader))
        for data in t:
            images = data['image'].to(device, dtype=torch.float)
            targets = data['target'].to(device, dtype=torch.float)

            batch_size = images.shape[0]

            # Forward pass
            outputs = model(images).squeeze()

            # Compute the loss
            loss = criterion(outputs, targets)

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            y_true.extend(targets.cpu().numpy())
            y_prob.extend(outputs.detach().cpu().numpy())

        epoch_loss = running_loss / dataset_size
        acc, f1, auc, pauc = compute_metrics(y_true, y_prob)

        print(f'Valid Epoch: {epoch}, Loss: {epoch_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, PAUC: {pauc:.4f}')

        gc.collect()

        return epoch_loss, acc, f1, auc, pauc     
    
    def train(self, model, train_loader, val_loader, num_accum, fold, num_epochs):

        start = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch_pauc = 0.0

        for epoch in range(num_epochs):
            gc.collect()

            # Training
            train_loss, train_acc, train_f1, train_auc, train_pauc = self.train_one_epoch(model, self.optimizer, self.criterion, self.scheduler, train_loader, self.device, num_accum, epoch)

            # Save training history
            self.append_to_history({'loss': train_loss, 'acc': train_acc, 'f1': train_f1, 'auc': train_auc, 'pauc': train_pauc}, 'train')
            self.history['train']['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.history['train']['epoch'].append(epoch)

            # Validation
            if val_loader is not None:
                with torch.no_grad():
                    val_loss, val_acc, val_f1, val_auc, val_pauc = self.valid_one_epoch(model, self.criterion, val_loader, self.device, epoch)
                
                # Save validation history
                self.append_to_history({'loss': val_loss, 'acc': val_acc, 'f1': val_f1, 'auc': val_auc, 'pauc': val_pauc}, 'val')
                self.history['val']['epoch'].append(epoch)

                # Save best model
                if best_epoch_pauc <= val_pauc:
                    print(f'Validation PAUC improved from {best_epoch_pauc:.4f} to {val_pauc:.4f}!')
                    best_epoch_pauc = val_pauc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_path = 'pauc_{:.4f}_Loss{:.4f}_epoch_{}.pth'.format(val_pauc, val_loss, epoch)
                else:
                    print(f'Validation PAUC did not improve!')
            

        end = time.time()

        time_elapsed = end - start

        fold_folder = self.weights_folder

        if fold is not None:
            print('Training of fold {} complete in {:.0f} {:.0f}m {:.0f}s'.format(fold, time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
            print('Best val PAUC: {:4f}'.format(best_epoch_pauc))
            
            fold_folder = os.path.join(self.weights_folder, 'fold_{}'.format(fold))
            if not os.path.exists(fold_folder):
                os.makedirs(fold_folder)
                
            # Save best model        
            torch.save(best_model_wts, os.path.join(fold_folder, best_model_path))
            print('Best model saved at: ', os.path.join(fold_folder, best_model_path))

        # Save last model
        last_model_path = 'pauc_{:.4f}_Loss{:.4f}_epoch_{}.pth'.format(val_pauc, val_loss, epoch) if val_loader is not None else 'final_model.pth'
        torch.save(model.state_dict(), os.path.join(fold_folder, last_model_path))

        