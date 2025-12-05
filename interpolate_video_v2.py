import torch
import numpy as np
import pandas as pd
import os
import subprocess
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from inference_cfm_v2 import CFMSamplerV2

def interpolate_video_v2(args):
    # 1. Initialize Sampler V2
    sampler = CFMSamplerV2(args.checkpoint, args.scaler)
    
    # 2. Load Real Data (for overlay)
    print(f"Loading data from {args.data}...")
    if args.data.endswith('.pkl'):
        df = pd.read_pickle(args.data)
    else:
        df = pd.read_csv(args.data)
        
    # Filter for RPM/Ball to identify "Available Real Fills"
    tol = 1e-5
    subset = df[
        (np.abs(df['rpm'] - args.rpm) < tol) & 
        (np.abs(df['ball_size'] - args.ball) < tol)
    ]
    real_fills = sorted(subset['fill_ratio'].unique())
    print(f"Real Data available at fills: {real_fills}")
    
    # 3. Define Interpolation Range
    min_fill = min(real_fills) if real_fills else 0.06
    max_fill = max(real_fills) if real_fills else 0.3
    
    n_frames = 100
    sweep_fills = np.linspace(min_fill, max_fill, n_frames)
    
    print(f"Generating {n_frames} interpolated frames from Fill {min_fill} to {max_fill}...")
    
    os.makedirs("frames_interp_v2", exist_ok=True)
    subprocess.run("rm frames_interp_v2/*.png", shell=True)
    
    # 4. Frame Generation Loop
    for i, curr_fill in enumerate(sweep_fills):
        print(f"Frame {i+1}/{n_frames} (Fill={curr_fill:.3f})...", end='\r')
        
        # A. Check for Real Data Match
        match_fill = None
        for rf in real_fills:
            if abs(curr_fill - rf) < 0.002: # Tolerance
                match_fill = rf
                break
        
        # B. Generate AI Data (V2: Flux + Collisions)
        syn_vals = sampler.sample(args.rpm, curr_fill, args.ball, num_samples=2000)
        syn_flux_n = syn_vals[:, 0]
        syn_flux_s = syn_vals[:, 1]
        syn_cols = syn_vals[:, 2]
        
        # Calculate Syn Metrics
        total_e_syn = np.sum(syn_flux_n + syn_flux_s)
        total_c_syn = np.sum(syn_cols)
        if total_c_syn < 1e-9: total_c_syn = 1e-9
        
        epc_syn = total_e_syn / total_c_syn
        enpc_syn = np.sum(syn_flux_n) / total_c_syn
        espc_syn = np.sum(syn_flux_s) / total_c_syn
        
        # C. Prepare Real Data (If matched)
        real_flux_n = None
        real_flux_s = None
        real_metrics_txt = ""
        
        if match_fill is not None:
            real_sub = subset[np.abs(subset['fill_ratio'] - match_fill) < tol].copy()
            if 'cum_normal_energy' in real_sub.columns:
                real_sub = real_sub.sort_values('time')
                real_sub['flux_normal'] = real_sub['cum_normal_energy'].diff()
                real_sub['flux_shear'] = real_sub['cum_shear_energy'].diff()
                real_sub = real_sub.dropna()
                real_sub = real_sub[(real_sub['flux_normal'] > 0)]
                real_sub = real_sub[real_sub['time'] > 5.1]
                
                real_flux_n = real_sub['flux_normal'].values
                real_flux_s = real_sub['flux_shear'].values
                real_cols_vals = real_sub['collisions'].values
                
                # Calculate Real Metrics
                tot_e_real = np.sum(real_flux_n + real_flux_s)
                tot_c_real = np.sum(real_cols_vals)
                if tot_c_real < 1e-9: tot_c_real = 1e-9
                
                epc_real = tot_e_real / tot_c_real
                enpc_real = np.sum(real_flux_n) / tot_c_real
                espc_real = np.sum(real_flux_s) / tot_c_real
                
                err = abs(epc_syn - epc_real) / epc_real * 100
                
                real_metrics_txt = (
                    f"\nReal: {epc_real:.2e} (N:{enpc_real:.2e}, S:{espc_real:.2e})\n"
                    f"Err: {err:.1f}%"
                )

        # D. Plotting
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: KDE
        axs[0].set_title(f"Flux Distribution (KDE)\nRPM={args.rpm}, Fill={curr_fill:.3f}, Ball={args.ball}mm")
        sns.kdeplot(x=syn_flux_n, y=syn_flux_s, fill=True, cmap="Oranges", alpha=0.5, thresh=0.05, levels=20, ax=axs[0])
        
        if real_flux_n is not None:
            idx = np.random.choice(len(real_flux_n), min(2000, len(real_flux_n)), replace=False)
            axs[0].scatter(real_flux_n[idx], real_flux_s[idx], s=10, alpha=0.3, color="blue", label="Real Data")
            axs[0].text(0.05, 0.95, "MATCHING REAL DATA", transform=axs[0].transAxes, 
                       color='blue', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
            
        axs[0].set_xlabel("Normal Flux")
        axs[0].set_ylabel("Shear Flux")
        axs[0].legend(loc='upper right')
        axs[0].grid(alpha=0.3)
        
        # Right: Cumulative
        axs[1].set_title("Cumulative Energy Growth")
        syn_time = np.arange(len(syn_flux_n)) * 0.01
        axs[1].plot(syn_time, np.cumsum(syn_flux_n), color='orange', label='Syn Normal')
        axs[1].plot(syn_time, np.cumsum(syn_flux_s), color='red', label='Syn Shear')
        
        if real_flux_n is not None:
            real_t = np.arange(len(real_flux_n)) * 0.01
            axs[1].plot(real_t, np.cumsum(real_flux_n), color='blue', linestyle='--', label='Real Normal')
            axs[1].plot(real_t, np.cumsum(real_flux_s), color='cyan', linestyle='--', label='Real Shear')
            
        # Stats Box (Position 0.65, 0.85 as requested)
        txt = (
            f"E/Col (J)\n"
            f"Syn: {epc_syn:.2e} (N:{enpc_syn:.2e}, S:{espc_syn:.2e})"
            f"{real_metrics_txt}"
        )
        axs[1].text(0.65, 0.85, txt, transform=axs[1].transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
                    
        axs[1].legend()
        axs[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"frames_interp_v2/frame_{i:03d}.png", dpi=150)
        plt.close(fig)

    print("\nStitching video...")
    cmd = f"ffmpeg -y -framerate 10 -i frames_interp_v2/frame_%03d.png -c:v libx264 -crf 15 -pix_fmt yuv420p {args.output}"
    subprocess.run(cmd, shell=True)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="cfm_checkpoints_v2/best_model_v2.pth")
    parser.add_argument("--data", type=str, default="extracted_energy_data_v2.pkl")
    parser.add_argument("--scaler", type=str, default="scaler_params_v2.pkl")
    parser.add_argument("--rpm", type=float, default=600)
    parser.add_argument("--ball", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="interpolation_sweep_v2.mp4")
    
    args = parser.parse_args()
    interpolate_video_v2(args)
