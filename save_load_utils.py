import numpy as np
import os

def save_prediction_data(unique_grids, final_prediction_array, base_path=None):
    """
    Save the unique grids and final prediction array to files.
    
    Args:
        unique_grids: numpy array of shape (N, 7, 7) containing grid designs
        final_prediction_array: numpy array containing predictions for each advisor
        base_path: optional path to save the files (defaults to current directory)
    """
    if base_path is None:
        base_path = os.getcwd()
    
    # Create a 'saved_data' directory if it doesn't exist
    save_dir = os.path.join(base_path, 'saved_data')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the arrays
    np.save(os.path.join(save_dir, 'unique_grids.npy'), unique_grids)
    np.save(os.path.join(save_dir, 'final_prediction_array.npy'), final_prediction_array)

def load_prediction_data(base_path=None):
    """
    Load the unique grids and final prediction array from files.
    
    Args:
        base_path: optional path where the files are saved (defaults to current directory)
    
    Returns:
        tuple: (unique_grids, final_prediction_array)
    """
    if base_path is None:
        base_path = os.getcwd()
    
    save_dir = os.path.join(base_path, 'saved_data')
    
    # Load the arrays
    unique_grids = np.load(os.path.join(save_dir, 'unique_grids.npy'))
    final_prediction_array = np.load(os.path.join(save_dir, 'final_prediction_array.npy'))
    
    return unique_grids, final_prediction_array