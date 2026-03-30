import pandas as pd
import numpy as np

def load_and_prep_data(file_path):
    # Define column names based on NASA CMAPSS documentation
    # The first two columns are ID and Time
    index_names = ['unit_nr', 'time_cycles']
    # The next three are operational settings
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    # The rest are sensor readings (s_1 to s_21)
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)] 
    
    col_names = index_names + setting_names + sensor_names
    
    # Read the data file (space-separated values)
    # Ensure header is None as the raw file has no headers
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)

    # --- Step 1: Calculate Remaining Useful Life (RUL) ---
    # Logic: Since this is run-to-failure data, RUL = Max Cycle - Current Cycle
    
    # Group by engine ID (unit_nr) to find the maximum cycle for each engine
    max_cycle = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_life']
    
    # Merge the max_life column back into the original dataframe
    df = df.merge(max_cycle, on='unit_nr', how='left')
    
    # Calculate RUL column
    df['RUL'] = df['max_life'] - df['time_cycles']
    
    # We limit the maximum RUL to 125. 
    # This teaches the model: "Anything above 125 is just 'Healthy'"
    # This is a standard technique for C-MAPSS dataset.
    df['RUL'] = df['RUL'].clip(upper=125)
    
    # --- Step 2: Drop Useless Columns ---
    # Based on data analysis of FD001, some sensors have constant values (zero variance)
    # These sensors do not contribute to the model and should be removed
    # Sensors to drop: s_1, s_5, s_6, s_10, s_16, s_18, s_19
    # Also drop setting_3 as it is constant in FD001
    cols_to_drop = ['setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19', 'max_life']
    
    df_clean = df.drop(columns=cols_to_drop)
    
    return df_clean

if __name__ == "__main__":
    # Path to your dataset file
    file_path = r'D:\study\Uni_Matrial\Final_Project\DigitalTwin\CMAPSSData\train_FD001.txt'
    
    try:
        print("Loading and processing data...")
        train_df = load_and_prep_data(file_path)
        
        # Display the shape of the processed data
        print(f"Data Shape: {train_df.shape}")
        
        # Show the first few rows to verify RUL calculation
        print("\nFirst 5 rows (High RUL):")
        print(train_df[['unit_nr', 'time_cycles', 'RUL']].head())
        
        # Show the last few rows to verify failure point (RUL near 0)
        print("\nLast 5 rows (Low RUL):")
        print(train_df[['unit_nr', 'time_cycles', 'RUL']].tail())
        
        # Save the processed data to a CSV file for the training step
        output_file = 'train_FD001_processed.csv'
        train_df.to_csv(output_file, index=False)
        print(f"\n Processed data saved to '{output_file}'")
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")