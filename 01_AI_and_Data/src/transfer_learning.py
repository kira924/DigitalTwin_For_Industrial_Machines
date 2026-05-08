import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf

# --- Configuration ---
seq_length = 30 

cols_to_drop = ['setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)] 
all_col_names = index_names + setting_names + sensor_names

def process_data(df, is_train=True, scaler=None):
    df_clean = df.drop(columns=cols_to_drop)
    feature_cols = [c for c in df_clean.columns if c not in ['unit_nr', 'time_cycles', 'RUL', 'max_life']]
    
    if is_train:
        # We need a NEW scaler for the new machine because its range might be different
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])
        return df_clean, scaler, feature_cols
    else:
        df_clean[feature_cols] = scaler.transform(df_clean[feature_cols])
        return df_clean, feature_cols

def gen_sequence(id_df, seq_len, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements - seq_len), range(seq_len, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_len, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_len:num_elements, :]

if __name__ == "__main__":
    print("Starting Transfer Learning Experiment...")
    
    # 1. Load the New Data (FD002 - The Complex Customer Machine)
    print("Loading New Dataset (FD002)...")
    train_FD002 = r'D:\study\Uni_Matrial\Final_Project\DigitalTwin\CMAPSSData\train_FD002.txt'
    try:
        new_df = pd.read_csv(train_FD002, sep='\s+', header=None, names=all_col_names)
    except FileNotFoundError:
        print("Error: train_FD002.txt not found. Please add it to the folder.")
        exit()

    # Prep RUL for FD002
    max_cycle = new_df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_life']
    new_df = new_df.merge(max_cycle, on='unit_nr', how='left')
    new_df['RUL'] = new_df['max_life'] - new_df['time_cycles']
    new_df['RUL'] = new_df['RUL'].clip(upper=125)
    new_df.drop(columns=['max_life'], inplace=True)
    
    # Simulate "Small Data": We only have data for the first 20 engines of this new type
    # This proves we can learn with LESS data using Transfer Learning
    small_new_df = new_df[new_df['unit_nr'] <= 20].copy()
    print(f"Using only {len(small_new_df['unit_nr'].unique())} engines for Fine-Tuning.")
    
    # Process Data
    train_df, new_scaler, feature_cols = process_data(small_new_df, is_train=True)
    
    seq_gen = (list(gen_sequence(train_df[train_df['unit_nr'] == id], seq_length, feature_cols)) 
               for id in train_df['unit_nr'].unique())
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    
    label_gen = [gen_labels(train_df[train_df['unit_nr'] == id], seq_length, ['RUL']) 
                 for id in train_df['unit_nr'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)

    # 2. Load the Pre-trained Brain (FD001 Model)
    print("\nLoading Pre-trained Model (FD001)...")
    rul_v2_model = r'D:\study\Uni_Matrial\Final_Project\DigitalTwin\RUL\rul_v2_model.keras'
    base_model = load_model(rul_v2_model)
    
    # 3. Apply Transfer Learning Strategy
    # Freeze the LSTM layers (Keep the "knowledge" of how engines work generally)
    # We assume the first 2 layers are LSTM layers based on our previous architecture

    base_model.layers[0].trainable = False  # Freeze input processing

    for layer in base_model.layers[1:]:     # Unfreeze the rest
        layer.trainable = True
        
    print(f"Layer {base_model.layers[0].name} frozen. Others represent 'Plasticity'.")

    # Re-compile with a slightly higher learning rate than before, but still low
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    base_model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    base_model.summary() # You should see more Trainable params now

    # 4. Fine-Tune (Increase Epochs)
    print("\nFine-Tuning on New Machine Data...")
    # Increased epochs to 40 because FD002 is harder
    history = base_model.fit(seq_array, label_array, epochs=40, batch_size=32, validation_split=0.1, verbose=1)
    # 5. Save the "Specialized" Model
    base_model.save('rul_fd002_finetuned.keras')
    print(" New Specialized Model Saved: rul_fd002_finetuned.keras")
    
    # 6. Quick Test on FD002 Test Data (Engine 10)
    print("\nTesting on FD002 (Engine 10)...")
    test_FD002 = r'D:\study\Uni_Matrial\Final_Project\DigitalTwin\CMAPSSData\test_FD002.txt'
    test_df = pd.read_csv(test_FD002, sep='\s+', header=None, names=all_col_names)
    RUL_FD002 = r'D:\study\Uni_Matrial\Final_Project\DigitalTwin\CMAPSSData\RUL_FD002.txt'
    test_rul_df = pd.read_csv(RUL_FD002, sep='\s+', header=None, names=['actual_rul'])
    
    test_df, _ = process_data(test_df, is_train=False, scaler=new_scaler)
    
    engine_id = 10
    engine_data = test_df[test_df['unit_nr'] == engine_id]
    data_matrix = engine_data[feature_cols].values
    
    sequences = []
    for i in range(len(data_matrix) - seq_length + 1):
        sequences.append(data_matrix[i:i+seq_length])
        
    if len(sequences) > 0:
        sequences = np.array(sequences)
        preds = base_model.predict(sequences)
        
        # Plot
        final_true_rul = test_rul_df.iloc[engine_id - 1]['actual_rul']
        time_steps = np.arange(len(preds))
        calculated_true_rul = [final_true_rul + len(preds) - i for i in range(len(preds))]
        clipped_true_rul = [min(125, x) for x in calculated_true_rul]
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, clipped_true_rul, label='Actual RUL', color='green', linestyle='--')
        plt.plot(time_steps, preds, label='Fine-Tuned Prediction', color='blue', linewidth=2)
        plt.title(f'Transfer Learning Result - FD002 Engine {engine_id}', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('transfer_learning_result.png')
        print("Check 'transfer_learning_result.png'")
        plt.show()