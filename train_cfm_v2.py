import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import argparse
from tqdm import tqdm

# Import V2 Modules
from cfm_dataset_v2 import FluxDatasetV2
from cfm_model_v2 import SimpleResNetV2

def train(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Prepare Data
    print("Loading Dataset V2...")
    full_dataset = FluxDatasetV2(args.data, scaler_path=args.scaler_out, fit_scalers=True)
    
    # Validation Split Strategy (Hold out Fill Ratios)
    df = full_dataset.processed_df
    unique_fills = df['fill_ratio'].unique()
    
    # Try to hold out 0.06 and 0.2 if present
    desired_val_fills = [0.06, 0.2]
    val_fills = [f for f in desired_val_fills if f in unique_fills]
    
    # Fallback if specific fills missing
    if len(val_fills) == 0:
        np.random.seed(42)
        val_fills = np.random.choice(unique_fills, size=max(1, int(len(unique_fills)*0.2)), replace=False)
        
    print(f"Holding out Fill Ratios for Validation: {val_fills}")
    
    val_indices = df[df['fill_ratio'].isin(val_fills)].index.tolist()
    train_indices = df[~df['fill_ratio'].isin(val_fills)].index.tolist()
    
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    
    # 3. Initialize Model V2
    model = SimpleResNetV2(
        dim_input=3, # Normal, Shear, Collisions
        dim_cond=3,  # RPM, Fill, Ball
        dim_hidden=512, 
        num_layers=8
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    
    # 4. Training Loop (OT-CFM)
    sigma = 0.0 # OT-CFM typically uses min-variance (sigma=0)
    
    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss_accum = 0
        
        for x1, c in train_loader:
            x1 = x1.to(device) # Real Data (Target)
            c = c.to(device)
            
            # Sample x0 (Gaussian Noise)
            x0 = torch.randn_like(x1).to(device)
            
            # Sample t (Uniform [0, 1])
            t = torch.rand(x1.shape[0], device=device)
            
            # Linear Interpolation (OT Path)
            # x_t = (1 - t) * x0 + t * x1
            # But t needs shape [B, 1]
            t_expand = t.view(-1, 1)
            psi_t = (1 - t_expand) * x0 + t_expand * x1
            
            # Target Velocity Field (v_t = x1 - x0)
            target_v = x1 - x0
            
            # Predict Velocity
            pred_v = model(t, psi_t, c)
            
            # Loss (MSE)
            loss = torch.mean((pred_v - target_v)**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for x1, c in val_loader:
                x1 = x1.to(device)
                c = c.to(device)
                x0 = torch.randn_like(x1)
                t = torch.rand(x1.shape[0], device=device)
                t_expand = t.view(-1, 1)
                psi_t = (1 - t_expand) * x0 + t_expand * x1
                target_v = x1 - x0
                pred_v = model(t, psi_t, c)
                loss = torch.mean((pred_v - target_v)**2)
                val_loss_accum += loss.item()
                
        avg_val_loss = val_loss_accum / len(val_loader)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.checkpoint)
            # print(f"Saved Best Model (Val Loss: {best_val_loss:.6f})")

    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="extracted_energy_data_v2.pkl")
    parser.add_argument("--checkpoint", type=str, default="cfm_checkpoints_v2/best_model_v2.pth")
    parser.add_argument("--scaler_out", type=str, default="scaler_params_v2.pkl")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    train(args)
