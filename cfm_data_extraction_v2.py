import os
import glob
import re
import numpy as np
import pandas as pd
import pickle
from edempy import Deck

def parse_conditions_from_path(path):
    """
    Parses RPM, Fill Ratio, and Ball Size from the directory path.
    Heuristic:
    - Decimal value -> Fill Ratio
    - Value with 'mm' -> Ball Size
    - Value without units (integer-like) -> RPM
    
    Splits path by separators and looks for these patterns in the folder names.
    Returns a dictionary of conditions.
    """
    # Normalize path and split into components
    path = os.path.normpath(path)
    parts = path.split(os.sep)
    
    # We look at the last few folders. The .dem file is usually in a folder named after the conditions 
    # or the conditions are in the hierarchy.
    
    conditions = {
        "rpm": None,
        "fill_ratio": None,
        "ball_size": None
    }
    
    # Helper to clean strings
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    # Iterate through parts in reverse order (closer to file)
    for part in reversed(parts):
        # Split part by underscores or spaces if multiple params are in one folder name
        tokens = re.split(r'[_\s]+', part)
        
        for token in tokens:
            token = token.lower()
            
            # Check for Ball Size (contains 'mm')
            if 'mm' in token:
                # Extract number before mm
                match = re.search(r'(\d*\.?\d+)mm', token)
                if match:
                    conditions["ball_size"] = float(match.group(1))
                    continue
            
            # Check for Fill Ratio (decimal value, usually < 1.0 but not strictly)
            # We assume RPM is integer-like (e.g. 300) and Fill Ratio is float-like (e.g. 0.2, 0.25)
            # However, RPM could be float 300.0. 
            # Context from user: "only value with decimals is the fill ratio"
            if is_float(token):
                val = float(token)
                if '.' in token:
                     # likely fill ratio
                     if conditions["fill_ratio"] is None:
                         conditions["fill_ratio"] = val
                else:
                    # likely RPM (no decimal point)
                    if conditions["rpm"] is None:
                        conditions["rpm"] = val
            
            # Stop if all found? 
            # Sometimes there are multiple numbers (dates, ids). 
            # We rely on the user's description being the primary differentiator.
            
    return conditions

def extract_energy_from_deck(deck_path, target_dt=0.01):
    print(f"Processing deck: {deck_path}")
    try:
        deck = Deck(deck_path)
    except Exception as e:
        print(f"Failed to load deck {deck_path}: {e}")
        return None

    # Get conditions from path
    conditions = parse_conditions_from_path(os.path.dirname(deck_path))
    print(f"Extracted conditions: {conditions}")

    if any(v is None for v in conditions.values()):
        print("Warning: Could not fully identify conditions (RPM, Fill Ratio, Ball Size) from path.")

    # Property names mapping
    # Note: These names must match exactly what's in the EDEM deck. 
    # Based on previous script: 'Particle-Wall Normal Energy Loss', etc.
    prop_names = deck.creatorData.simulationCustomPropertyNames
    
    def get_prop_index(name_list, pattern):
        for i, name in enumerate(name_list):
            if pattern in name:
                return i
        return None

    # We need indices for Normal and Tangential Energy Loss (Particle-Wall and Particle-Particle)
    # The previous script summed PW and PP losses.
    
    idxs = {}
    patterns = {
        'pw_n': 'Particle-Wall Normal Energy Loss',
        'pp_n': 'Particle-Particle Normal Energy Loss',
        'pw_t': 'Particle-Wall Tangential Energy Loss',
        'pp_t': 'Particle-Particle Tangential Energy Loss'
    }
    
    for key, pat in patterns.items():
        idx = get_prop_index(prop_names, pat)
        if idx is not None:
            idxs[key] = idx
        else:
            print(f"Warning: Property '{pat}' not found in deck.")
    
    # Extract Data
    data_records = []
    
    num_timesteps = deck.numTimesteps
    timestep_values = deck.timestepValues # Array of times
    
    # Find indices corresponding to target_dt spacing
    # We want t=0, t=dt, t=2dt...
    # We scan the available timesteps.
    
    current_target_time = 0.0
    
    for t in range(num_timesteps):
        time_t = timestep_values[t]
        
        # Simple sampling: if time_t is >= current_target_time, take it and increment target
        
        if time_t >= current_target_time:
            # Extract
            # Get Cumulative Energies
            try:
                # Normal
                e_pw_n = deck.timestep[t].customProperties[idxs['pw_n']].getData()[0] if 'pw_n' in idxs else 0.0
                e_pp_n = deck.timestep[t].customProperties[idxs['pp_n']].getData()[0] if 'pp_n' in idxs else 0.0
                cum_normal = e_pw_n + e_pp_n
                
                # Shear / Tangential
                e_pw_t = deck.timestep[t].customProperties[idxs['pw_t']].getData()[0] if 'pw_t' in idxs else 0.0
                e_pp_t = deck.timestep[t].customProperties[idxs['pp_t']].getData()[0] if 'pp_t' in idxs else 0.0
                cum_shear = e_pw_t + e_pp_t
                
                # --- NEW: Extract Collision Counts ---
                # "numcol_SG = deck.timestep[t].collision.surfGeom.getNumCollisions()"
                # "numcol_SS = deck.timestep[t].collision.surfSurf.getNumCollisions()"
                
                # Note: edempy access might fail if collisions not processed or different API version.
                # Assuming standard API as provided by user.
                try:
                    numcol_sg = deck.timestep[t].collision.surfGeom.getNumCollisions()
                except:
                    numcol_sg = 0
                
                try:
                    numcol_ss = deck.timestep[t].collision.surfSurf.getNumCollisions()
                except:
                    numcol_ss = 0
                    
                total_collisions = numcol_sg + numcol_ss
                
                record = {
                    'time': time_t,
                    'cum_normal_energy': cum_normal,
                    'cum_shear_energy': cum_shear,
                    'collisions': total_collisions, # New Field
                    'rpm': conditions['rpm'],
                    'fill_ratio': conditions['fill_ratio'],
                    'ball_size': conditions['ball_size'],
                    'source_deck': os.path.basename(deck_path)
                }
                data_records.append(record)
                
                # Advance target time
                current_target_time += target_dt
                
            except Exception as e:
                print(f"Error extracting at step {t}: {e}")
                
    return pd.DataFrame(data_records)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract EDEM energy data for CFM training (v2 - with collisions).")
    parser.add_argument("--root", type=str, default="/Volumes/Extreme SSD/Paper 1/Final backup paper 1/Real simulations (windows edem)/First paper/Fill ratio variation", help="Root directory to search for .dem files.")
    parser.add_argument("--output", type=str, default="extracted_energy_data_v2.pkl", help="Output pickle filename.")
    parser.add_argument("--dt", type=float, default=0.01, help="Target timestep interval (seconds).")
    args = parser.parse_args()

    all_decks = []
    # Recursively find .dem files
    print(f"Searching for .dem files in {os.path.abspath(args.root)}...")
    for dirpath, dirnames, filenames in os.walk(args.root):
        for file in filenames:
            if file.endswith(".dem"):
                all_decks.append(os.path.join(dirpath, file))
    
    if not all_decks:
        print("No .dem files found in subdirectories.")
        return

    print(f"Found {len(all_decks)} deck files.")
    
    all_data = []
    
    for deck_path in all_decks:
        df = extract_energy_from_deck(deck_path, target_dt=args.dt)
        if df is not None and not df.empty:
            all_data.append(df)
            
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Sort by deck and time just in case
        final_df = final_df.sort_values(by=['source_deck', 'time'])
        
        print("Extraction complete.")
        print(final_df.head())
        print(f"Total records: {len(final_df)}")
        
        with open(args.output, 'wb') as f:
            pickle.dump(final_df, f)
        print(f"Saved to {args.output}")
        
        # Also save as CSV for inspection
        csv_name = args.output.replace('.pkl', '.csv')
        final_df.to_csv(csv_name, index=False)
        print(f"Saved to {csv_name}")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()

