import torch
import numpy as np

seed = 42

random_seed = False
if random_seed:
    seed = np.random.randint(1000)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)



## -------------------------------- ##
project = "toy"
input_path = "./inputs/" + project + "/"

## training dataset names
x_train_file = input_path + "train_x_" + project + ".npy"
y_train_file = input_path + "train_y_" + project + ".npy"

## test dataset
x_test_file = input_path + "test_x_" + project + ".npy"
y_test_file = input_path + "test_y_" + project + ".npy"


## -------------------------------- ##

# num of epochs
epochs = 5000
# batch size
batch_size = 2048



# output paths and state name
output_path = "./outputs/" + project + "/"
plots_path = "./plots/" + project + "/"
ckpt_path = "./model-state/" + project + "/"
state_name = 'final_state_b' + str(batch_size) + '_it' + str(epochs)

## Hyperparameter of diffusion model and perturbation process
lr = 3e-4
beta_1 = 1e-4
beta_T = 0.02
T = 500

n_dims = 4 # dimensions of the data vector
shape_in = (n_dims*2,)
shape_out = (n_dims,)

device = torch.device('cuda:0')

