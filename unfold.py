import torch
import tqdm
import os
import numpy as np
from model import Model
from diffusion import DiffusionProcess
import configs


configs.set_seed(configs.seed)

if not os.path.exists(configs.output_path):
   os.makedirs(configs.output_path)


print("loading data...")
data_y = np.load(configs.y_test_file, mmap_mode='r')
data_y = torch.from_numpy(np.array(data_y, copy=True)).float().to(configs.device)

# cleanup for loading saved state from compiled model
state_dict = torch.load(configs.ckpt_path + configs.state_name + '.pth', weights_only=True)
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

# initiate models and load saved state
cDDPM = Model(configs.device, configs.beta_1, configs.beta_T, configs.T, configs.shape_in[0], configs.shape_out[0])
cDDPM.load_state_dict(state_dict)
process = DiffusionProcess(configs.beta_1, configs.beta_T, configs.T, cDDPM, configs.device, configs.shape_out,)

print("sampling...")

n = 100
# sample size during inference
sample_batch_size = data_y.shape[0]//n
pbar = tqdm.tqdm(total=n)

pred_x = []
with torch.no_grad():
   for i in range(0, data_y.shape[0], sample_batch_size):
       y_part = data_y[i:i+sample_batch_size]
       pred_x.append(process.sampling(y_part.shape[0], y_part))

pred_x = torch.cat(pred_x)
pred_x = pred_x.cpu().numpy()

## save pred_x results
np.save(configs.output_path + "pred_x_" + configs.project + ".npy", pred_x)

print("done!")

