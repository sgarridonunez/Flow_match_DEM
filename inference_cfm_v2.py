import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import seaborn as sns
from torchdiffeq import odeint

# Import V2 Modules
from cfm_model_v2 import SimpleResNetV2
from cfm_dataset_v2 import FluxDatasetV2

class CFMSamplerV2:
    def __init__(self, checkpoint_path, scaler_path="scaler_params_v2.pkl", device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # 1. Load Scalers
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            self.flux_scaler = scalers['flux_scaler']
            self.cond_scaler = scalers['cond_scaler']
            
        # 2. Load Model V2 (3D Input)
        self.model = SimpleResNetV2(dim_input=3, dim_cond=3, dim_hidden=512, num_layers=8).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"Loaded model V2 from {checkpoint_path}")

    def sample(self, rpm, fill_ratio, ball_size, num_samples=2000):
        """
        Generates synthetic samples: [Normal Flux, Shear Flux, Collision Count]
        """
        # 1. Condition
        cond_raw = np.array([[rpm, fill_ratio, ball_size]])
        cond_norm = self.cond_scaler.transform(cond_raw)
        c = torch.tensor(cond_norm, dtype=torch.float32, device=self.device)
        c = c.repeat(num_samples, 1)
        
        # 2. Noise x0 (3D)
        x0 = torch.randn(num_samples, 3, device=self.device)
        
        # 3. ODE Wrapper
        class ODEFunc(nn.Module):
            def __init__(self, model, c):
                super().__init__()
                self.model = model
                self.c = c
            def forward(self, t, x):
                t_batch = torch.ones(x.shape[0], device=x.device) * t
                return self.model(t_batch, x, self.c)
        
        ode_func = ODEFunc(self.model, c)
        
        # 4. Integrate
        t_span = torch.tensor([0.0, 1.0], device=self.device)
        with torch.no_grad():
            traj = odeint(ode_func, x0, t_span, method='dopri5', rtol=1e-4, atol=1e-4)
            
        # 5. Inverse Transform
        x1 = traj[-1].cpu().numpy()
        
        # [Normal, Shear, Collisions]
        real_log = self.flux_scaler.inverse_transform(x1)
        real_vals = np.expm1(real_log)
        
        return real_vals

    def plot_comparison(self, real_data_path, rpm, fill_ratio, ball_size, save_path=None):
        """
        Comparison Plot V2:
        Left: Flux KDE
        Right: Cumulative Energy
        Metrics: Includes Energy per Collision!
        """
        # Load Real Data logic (simplified)
        if isinstance(real_data_path, str):
            if real_data_path.endswith('.pkl'):
                df = pd.read_pickle(real_data_path)
            else:
                df = pd.read_csv(real_data_path)
        else:
            df = real_data_path

        # Filter
        tol = 1e-5
        subset = df[
            (np.abs(df['rpm'] - rpm) < tol) & 
            (np.abs(df['fill_ratio'] - fill_ratio) < tol) & 
            (np.abs(df['ball_size'] - ball_size) < tol)
        ].copy()
        
        real_data_available = len(subset) > 0
        
        if real_data_available:
            # Process Real Data
            # Note: We need collisions too
            if 'cum_normal_energy' in subset.columns:
                subset = subset.sort_values('time')
                subset['flux_normal'] = subset['cum_normal_energy'].diff()
                subset['flux_shear'] = subset['cum_shear_energy'].diff()
                subset = subset.dropna()
                subset = subset[(subset['flux_normal'] > 0)] # Filter noise
                subset = subset[subset['time'] > 5.1]
                
                real_flux_n = subset['flux_normal'].values
                real_flux_s = subset['flux_shear'].values
                real_cols = subset['collisions'].values
                real_time = subset['time'].values
                if len(real_time) > 0: real_time -= real_time[0]
                
                n_syn = len(real_flux_n)
            else:
                 # Fallback
                 n_syn = 2000
        else:
            n_syn = 2000

        # Generate Synthetic
        print(f"Generating {n_syn} samples (V2)...")
        syn_vals = self.sample(rpm, fill_ratio, ball_size, num_samples=n_syn)
        syn_flux_n = syn_vals[:, 0]
        syn_flux_s = syn_vals[:, 1]
        syn_cols = syn_vals[:, 2] # Collision Counts
        
        # --- Derived Physics ---
        # 1. Total Power (Sum of Flux over all samples)
        # 2. Total Collisions (Sum of Counts over all samples)
        # 3. Energy per Collision = Total Power / Total Collisions
        
        total_flux_n_syn = np.sum(syn_flux_n)
        total_flux_s_syn = np.sum(syn_flux_s)
        total_cols_syn = np.sum(syn_cols)
        
        # Avoid divide by zero
        if total_cols_syn < 1e-9: total_cols_syn = 1e-9
        
        e_per_col_syn = (total_flux_n_syn + total_flux_s_syn) / total_cols_syn
        e_n_per_col_syn = total_flux_n_syn / total_cols_syn
        e_s_per_col_syn = total_flux_s_syn / total_cols_syn
        
        print(f"\n--- Physics V2 (Synthetic) ---")
        print(f"Total E/Col:  {e_per_col_syn:.6f} J")
        print(f"Normal E/Col: {e_n_per_col_syn:.6f} J")
        print(f"Shear E/Col:  {e_s_per_col_syn:.6f} J")
        
        if real_data_available:
            total_flux_n_real = np.sum(real_flux_n)
            total_flux_s_real = np.sum(real_flux_s)
            total_cols_real = np.sum(real_cols)
            if total_cols_real < 1e-9: total_cols_real = 1e-9
            
            e_per_col_real = (total_flux_n_real + total_flux_s_real) / total_cols_real
            e_n_per_col_real = total_flux_n_real / total_cols_real
            e_s_per_col_real = total_flux_s_real / total_cols_real
            
            print(f"\n--- Physics V2 (Real) ---")
            print(f"Total E/Col:  {e_per_col_real:.6f} J")
            print(f"Normal E/Col: {e_n_per_col_real:.6f} J")
            print(f"Shear E/Col:  {e_s_per_col_real:.6f} J")
            
            err = abs(e_per_col_syn - e_per_col_real) / e_per_col_real * 100
            print(f"\nTotal Error: {err:.2f}%")

        # --- Plotting ---
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Flux KDE
        axs[0].set_title(f"Flux Distribution (KDE)\nRPM={rpm}, Fill={fill_ratio}, Ball={ball_size}")
        sns.kdeplot(x=syn_flux_n, y=syn_flux_s, fill=True, cmap="Oranges", alpha=0.5, thresh=0.05, levels=20, ax=axs[0])
        if real_data_available:
            idx = np.random.choice(len(real_flux_n), min(2000, len(real_flux_n)), replace=False)
            axs[0].scatter(real_flux_n[idx], real_flux_s[idx], s=10, alpha=0.3, color="blue")
        axs[0].set_xlabel("Normal Flux")
        axs[0].set_ylabel("Shear Flux")

        # Right: Cumulative + Collision Text
        axs[1].set_title("Cumulative Energy")
        t_syn = np.arange(len(syn_flux_n)) * 0.01
        axs[1].plot(t_syn, np.cumsum(syn_flux_n), color='orange', label='Syn Normal')
        axs[1].plot(t_syn, np.cumsum(syn_flux_s), color='red', label='Syn Shear')
        
        if real_data_available:
            axs[1].plot(real_time, np.cumsum(real_flux_n), color='blue', linestyle='--', label='Real Normal')
            axs[1].plot(real_time, np.cumsum(real_flux_s), color='cyan', linestyle='--', label='Real Shear')
            
            # Add V2 Metrics Text
            txt = (
                f"E/Col (J)\n"
                f"Syn: {e_per_col_syn:.2e} (N:{e_n_per_col_syn:.2e}, S:{e_s_per_col_syn:.2e})\n"
                f"Real: {e_per_col_real:.2e} (N:{e_n_per_col_real:.2e}, S:{e_s_per_col_real:.2e})\n"
                f"Err: {err:.1f}%"
            )
            axs[1].text(0.65, 0.85, txt, transform=axs[1].transAxes, fontsize=9, 
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
            
        axs[1].legend()
        plt.tight_layout()
        
        if save_path is None: save_path = f"comparison_v2_rpm{rpm}_fill{fill_ratio}.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="cfm_checkpoints_v2/best_model_v2.pth")
    parser.add_argument("--data", type=str, default="extracted_energy_data_v2.pkl")
    parser.add_argument("--scaler", type=str, default="scaler_params_v2.pkl")
    parser.add_argument("--rpm", type=float, default=600)
    parser.add_argument("--fill", type=float, default=0.2)
    parser.add_argument("--ball", type=float, default=10.0)
    
    args = parser.parse_args()
    
    if os.path.exists(args.checkpoint):
        sampler = CFMSamplerV2(args.checkpoint, args.scaler)
        sampler.plot_comparison(args.data, args.rpm, args.fill, args.ball)
    else:
        print("Model V2 not found.")
