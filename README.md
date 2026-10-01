# Unfolding with Conditional Diffusion Models

This project uses a conditional denoising diffusion probabilistic model (cDDPM) to unfold detector-level (reco) data back to particle-level (truth). The model learns to generate truth events conditioned on their reco counterparts. At inference time, it produces an unfolded truth event for each reco event.

## Files

| File | Description |
|---|---|
| `configs.py` | Project name, file paths, training settings, and diffusion hyperparameters |
| `model.py` | MLP noise predictor and the `Model` wrapper that handles the forward (noising) process and loss |
| `diffusion.py` | `DiffusionProcess` class that runs the reverse (denoising) process to generate samples |
| `train.py` | Trains the model and saves the final state and a loss plot |
| `unfold.py` | Loads the trained model and unfolds the test reco data |


## Input data

Inputs go in `./inputs/<project>/` as NumPy arrays of shape `(n_events, n_dims)`:

```
train_x_<project>.npy   # truth, training
train_y_<project>.npy   # reco, training
test_x_<project>.npy    # truth, test (for validation of the unfolded result)
test_y_<project>.npy    # reco, test (input to the unfolding)
```

Row `i` of the `x` file must correspond to the same event as row `i` of the `y` file.

## Usage

1. Edit `configs.py`. Set `project` and `n_dims` to match your data.
2. Train:
   ```
   python train.py
   ```
   90% of the training data is used for training and 10% for validation.
3. Unfold:
   ```
   python unfold.py
   ```

## Outputs

| Path | Contents |
|---|---|
| `./model-state/<project>/` | Trained model state (`final_state_b<batch_size>_it<epochs>.pth`) |
| `./outputs/<project>/pred_x_<project>.npy` | Unfolded truth-level predictions |



## Requirements

- Python 3
- PyTorch 2.0 or later
- NumPy
- Matplotlib
- tqdm
- A CUDA GPU (set in `configs.device`)