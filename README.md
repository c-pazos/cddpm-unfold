# Unfolding with Conditional Diffusion Models


## Files

configs.py: File paths, training settings, and diffusion hyperparameters
model.py: MLP structure, Model class handles the forward (noising) process and loss
diffusion.py: DiffusionProcess class runs the reverse (denoising) process to generate samples
train.py: Trains the model and saves the final state and a loss plot
unfold.py: Loads the trained model and unfolds the test data


## Input data

Inputs should be arrays of shape (n_events, n_dims):

train_x_<project>.npy   # truth, training
train_y_<project>.npy   # reco, training
test_x_<project>.npy    # truth, test (for validation of the unfolded result)
test_y_<project>.npy    # reco, test (input to the unfolding)

Row i of the truth (x) file must correspond to the same event as row i of the reco (y) file.

## Usage

1. Edit `configs.py`
2. Train: python train.py
3. Unfold: python unfold.py

## Outputs
From train.py
./model-state/<project>/ -> trained model state

From unfold.py
./outputs/<project>/pred_x_<project>.npy -> unfolded truth-level predictions
