import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

class FluxDatasetV2(Dataset):
    def __init__(self, data_path, scaler_path="scaler_params_v2.pkl", fit_scalers=True):
        """
        Args:
            data_path: Path to extracted_energy_data_v2.pkl or .csv
            scaler_path: Where to save/load scaler parameters
            fit_scalers: If True, fits new scalers. If False, loads existing.
        """
        # Load Data
        if data_path.endswith('.pkl'):
            self.df = pd.read_pickle(data_path)
        else:
            self.df = pd.read_csv(data_path)
            
        self.scaler_path = scaler_path
        self._preprocess(fit_scalers)
        
    def _preprocess(self, fit_scalers):
        df = self.df.copy()
        
        # 1. Create Unique Identifier for Simulations
        grp_cols = ['rpm', 'fill_ratio', 'ball_size']
        if 'sim_id' not in df.columns:
            df['sim_id'] = df.groupby(grp_cols).ngroup()
            
        # 2. Compute Flux (Delta E) per simulation
        # Note: 'collisions' is instantaneous count, so we use the raw value.
        df = df.sort_values(by=['sim_id', 'time'])
        
        df['flux_normal'] = df.groupby('sim_id')['cum_normal_energy'].diff()
        df['flux_shear'] = df.groupby('sim_id')['cum_shear_energy'].diff()
        
        # 3. Cleaning Pipeline
        # Drop the first row of each group (NaNs from diff)
        df = df.dropna(subset=['flux_normal', 'flux_shear'])
        
        # Drop Startup Transient (First 5.1s)
        startup_time = 5.1
        len_before_startup = len(df)
        df = df[df['time'] > startup_time]
        print(f"Dropped {len_before_startup - len(df)} rows due to startup time < {startup_time}s.")
        
        # Drop Numerical Noise (Flux <= 0)
        total_flux = df['flux_normal'] + df['flux_shear']
        len_before_noise = len(df)
        df = df[total_flux > 1e-9] 
        print(f"Dropped {len_before_noise - len(df)} rows due to numerical noise (flux <= 0). Remaining: {len(df)}")
        
        # 4. Prepare Targets (x)
        # We want to predict [flux_normal, flux_shear, collisions]
        # Log Transform Flux (Heavy Tail)
        flux_normal = np.log1p(df['flux_normal'].values).reshape(-1, 1)
        flux_shear = np.log1p(df['flux_shear'].values).reshape(-1, 1)
        
        # Log Transform Collisions
        collisions = np.log1p(df['collisions'].values).reshape(-1, 1)
        
        x_raw = np.hstack([flux_normal, flux_shear, collisions])
        
        # 5. Prepare Conditions (c)
        rpm = df['rpm'].values.reshape(-1, 1)
        fill = df['fill_ratio'].values.reshape(-1, 1)
        ball = df['ball_size'].values.reshape(-1, 1)
        
        c_raw = np.hstack([rpm, fill, ball])
        
        # 6. Scaling
        if fit_scalers:
            print("Fitting new scalers...")
            self.flux_scaler = StandardScaler()
            self.x_norm = self.flux_scaler.fit_transform(x_raw)
            
            self.cond_scaler = MinMaxScaler()
            self.c_norm = self.cond_scaler.fit_transform(c_raw)
            
            # Save Scalers
            with open(self.scaler_path, 'wb') as f:
                pickle.dump({
                    'flux_scaler': self.flux_scaler,
                    'cond_scaler': self.cond_scaler
                }, f)
            print(f"Saved scalers to {self.scaler_path}")
            
        else:
            print(f"Loading scalers from {self.scaler_path}...")
            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.flux_scaler = scalers['flux_scaler']
                self.cond_scaler = scalers['cond_scaler']
                
            self.x_norm = self.flux_scaler.transform(x_raw)
            self.c_norm = self.cond_scaler.transform(c_raw)
            
        # Store processed data for referencing later if needed
        self.processed_df = df.reset_index(drop=True)
        
        # Convert to Tensor
        self.x = torch.tensor(self.x_norm, dtype=torch.float32)
        self.c = torch.tensor(self.c_norm, dtype=torch.float32)
        
        print(f"Dataset Ready. Shape x: {self.x.shape}, Shape c: {self.c.shape}")
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.c[idx]
