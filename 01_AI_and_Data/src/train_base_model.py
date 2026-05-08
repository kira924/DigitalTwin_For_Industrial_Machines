import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- Advanced Configuration ---
# Increased sequence length to capture longer historical patterns
SEQUENCE_LENGTH = 60 
BATCH_SIZE = 128      # Increased batch size for more stable gradient updates
EPOCHS = 50           # More epochs, but EarlyStopping will stop it if needed

def gen_sequence(id_df, seq_length, seq_cols):
    """
    Reshapes the DataFrame into a 3D array (Samples, Time Steps, Features)
    Only creates sequences if data length >= seq_length
    """
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
    """
    Extracts the target label (RUL) corresponding to the end of each sequence
    """
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

if __name__ == "__main__":
    # 1. Load Processed Data
    print("Loading data...")
    train_df = pd.read_csv('train_FD001_processed.csv')
    
    # 2. Normalize Data
    feature_cols = [col for col in train_df.columns if col not in ['unit_nr', 'time_cycles', 'RUL']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    
    print("Data normalized.")

    # 3. Prepare Sequences (Using the new length 50)
    # Note: Some engines might be shorter than 50 cycles, the code handles this by skipping them or loop logic
    seq_gen = (list(gen_sequence(train_df[train_df['unit_nr'] == id], SEQUENCE_LENGTH, feature_cols)) 
               for id in train_df['unit_nr'].unique())
    
    # Filter out empty sequences (in case an engine life < 50)
    seq_gen = [s for s in seq_gen if len(s) > 0]
    
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    
    # Generate Labels
    label_gen = [gen_labels(train_df[train_df['unit_nr'] == id], SEQUENCE_LENGTH, ['RUL']) 
                 for id in train_df['unit_nr'].unique()]
    
    # Filter empty labels
    label_gen = [l for l in label_gen if len(l) > 0]
    
    label_array = np.concatenate(label_gen).astype(np.float32)

    print(f"Input Shape (X): {seq_array.shape}")
    print(f"Target Shape (y): {label_array.shape}")

    # 4. Build a Deeper LSTM Model
    model = Sequential()
    
    # Layer 1: More units (128) to capture complex features
    model.add(LSTM(input_shape=(SEQUENCE_LENGTH, len(feature_cols)), units=128, return_sequences=True))
    model.add(Dropout(0.3)) # Higher dropout to prevent overfitting
    
    # Layer 2: Another LSTM layer
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.3))
    
    # Output Layer
    model.add(Dense(units=1, activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    
    # 5. Define Callbacks (The Secret Sauce)
    
    # Checkpoint: Save ONLY the best model (lowest validation loss)
    checkpoint = ModelCheckpoint(
        'rul_best_model.keras', 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min', 
        verbose=1
    )
    
    # ReduceLROnPlateau: If validation loss stops improving for 5 epochs, reduce learning rate
    # This helps the model get out of local minima
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,       # Multiply LR by 0.5
        patience=5,       # Wait 5 epochs before reducing
        min_lr=0.00001,   # Don't go below this
        verbose=1
    )
    
    # EarlyStopping: Stop training if no improvement for 10 epochs
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )

    print("\nStarting training with advanced callbacks...")
    history = model.fit(
        seq_array, 
        label_array, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.1, 
        verbose=1,
        callbacks=[checkpoint, reduce_lr, early_stop] # Add callbacks here
    )
    
    # Save the scaler for later use in simulation
    import joblib
    joblib.dump(scaler, 'scaler.gz')
    print("Scaler saved as 'scaler.gz'")