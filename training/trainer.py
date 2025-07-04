# training/trainer.py
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os
from typing import Dict, Optional
import numpy as np 

# class VAETrainer

# train_epoch

class VAETrainer: 
    # Trainer Class for VAE models. 
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader, 
                 lr: float = 1e-3,
                 device: torch.device = torch.device('cpu'),
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './logs'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = self.checkpoint_dir
        self.log_dir = log_dir

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)

        # lr scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', factor = 0.5, patience= 10)

        # write to tensorboard
        self.writer = SummaryWriter(log_dir)

        os.makedirs(checkpoint_dir, exist_ok= True)
        os.makedirs(log_dir, exist_ok= True)

        # training state 
        self.epoch = 0
        self.best_val_loss = float('inf') # FLTMAX

        def train_epoch(self) -> Dict[str, float]:
            # Single Epoch
            self.model.train()

            # as explained in ./math, 
            # total loss = reconstruction loss + kld loss
            total_loss = 0     
            total_recon_loss = 0
            total_kld_loss = 0

            # progress bar
            pbar = tqdm(self.train_loader, desc = f"Epoch {self.epoch}")

            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(self.device)

                # forward pass
                self.optimizer.zero_grad()
                outputs = self.model(data)

                # loss
                loss_dict = self.model.loss_function(**outputs)
                loss = loss_dict['loss']

                # backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_recon_loss += loss_dict['recon_loss'].item()
                total_kld_loss += loss_dict['kld_loss'].item()

                # update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Recon': f"{loss_dict['recon_loss'].item():.4f}",
                    'KL': f"{loss_dict['kld_loss'].item():.4f}"
                })

            avg_loss = total_loss / len(self.train_loader)
            avg_recon_loss = total_recon_loss / len(self.train_loader)
            avg_kld_loss = total_kld_loss / len(self.train_loader)

            return {
                'loss': avg_loss,
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kld_loss
            }
        
        def validate(self) -> Dict[str, float]:
            self.model.eval()
            total_loss = 0     
            total_recon_loss = 0
            total_kld_loss = 0

            with torch.no_grad():
                for data, _ in self.val_loader:
                    data = data.to(self.device)

                    # forward pass
                    outputs = self.model(data)
                    loss_dict = self.model.loss_function(**outputs)

                    # accumulate loss
                    total_loss += loss_dict['loss'].item()
                    total_recon_loss += loss_dict['recon_loss'].item()
                    total_kld_loss += loss_dict['kld_loss'].item()

            avg_loss = total_loss / len(self.val_loader)
            avg_recon_loss = total_recon_loss / len(self.val_loader)
            avg_kld_loss = total_kld_loss / len(self.val_loader)

            return {
                'loss': avg_loss,
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kld_loss
            }
        
        def train(self, num_epochs: int, save_every: int = 10):
            # Train the model for a specified number of epochs

            for epoch in range(num_epochs):
                self.epoch = epoch

                # training
                train_losses = self.train_epoch()

                # validate
                val_losses = self.validate()

                # Note: For ReduceLROnPlateau, call step() after validation and pass in the validation loss.
                # If you change the scheduler type (e.g., to StepLR), update this call accordingly.

                self.scheduler.step(val_losses['loss'])

                # log metrics
                self.writer.add_scalar('Loss/Train', train_losses['loss'], epoch)
                self.writer.add_scalar('Loss/Val', val_losses['loss'], epoch)
                self.writer.add_scalar('Recon_Loss/Train', train_losses['recon_loss'], epoch)
                self.writer.add_scalar('Recon_Loss/Val', val_losses['recon_loss'], epoch)
                self.writer.add_scalar('KL_Loss/Train', train_losses['kl_loss'], epoch)
                self.writer.add_scalar('KL_Loss/Val', val_losses['kl_loss'], epoch)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"Train Loss: {train_losses['loss']:.4f} | Val Loss: {val_losses['loss']:.4f}")
                print(f"Train Recon: {train_losses['recon_loss']:.4f} | Val Recon: {val_losses['recon_loss']:.4f}")
                print(f"Train KL: {train_losses['kl_loss']:.4f} | Val KL: {val_losses['kl_loss']:.4f}")

                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(epoch)

                # save the best model
                if val_losses['loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['loss']
                    self.save_checkpoint(epoch, is_best = True)

        def save_checkpoint(self, epoch: int, is_best: bool = False):

            checkpoint = {
                        'epoch': epoch, 
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'schecduler_state_dict': self.scheduler.state_dict(),
                        'best_val_loss': self.best_val_loss
            }       

            filename = f'checkpoint epoch {epoch}.pth'
            filepath = os.path.join(self.checkpoint_dir, filename)
            torch.save(checkpoint, filepath)

            if is_best:
                best_filepath = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save(checkpoint, best_filepath)
                print(f"SAVED BEST MODEL AT EPOCH {epoch}")

        def load_checkpoint(self, checkpoint_path: str):
            checkpoint = torch.load(checkpoint, map_location = self.device)
        
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
        
            print(f"Loaded checkpoint from epoch {self.epoch}")



