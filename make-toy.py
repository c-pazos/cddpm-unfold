import os
import argparse
import numpy as np

# usage:
#   python make_toy.py --seed 1 --name train
#   python make_toy.py --seed 2 --name test

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--name", type=str, default="train")  # "train" or "test"
parser.add_argument("--n", type=int, default=100000)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
n = args.n

out_path = "./inputs/toy/"
os.makedirs(out_path, exist_ok=True)


## -------- truth -------- ##

# x0: bimodal (two gaussians at 0.3 and 0.7), clipped to [0, 1]
peak = rng.choice([0.3, 0.7], size=n)
x0 = np.clip(rng.normal(peak, 0.08), 0, 1)

# x1: exponential truncated to [0, 1] (inverse CDF, so no clipping needed)
scale = 0.2
u = rng.uniform(0, 1, size=n)
x1 = -scale * np.log(1 - u * (1 - np.exp(-1 / scale)))

# x2: sum of the first two (range [0, 2])
x2 = x0 + x1

# x3: square of the first
x3 = x0**2

x = np.stack([x0, x1, x2, x3], axis=1).astype(np.float32)


## -------- smearing -------- ##

# different gaussian resolution for each component
sigmas = np.array([0.05, 0.03, 0.08, 0.02])
y = x + rng.normal(0, 1, size=x.shape) * sigmas
y = y.astype(np.float32)


## -------- save -------- ##

np.save(out_path + args.name + "_x_toy.npy", x)
np.save(out_path + args.name + "_y_toy.npy", y)

print(f"saved {n} events to {out_path} ({args.name}, seed {args.seed})")
