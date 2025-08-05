## Velocity Flow Matching: Compression and Dynamics Learning with Flow Matching! <br>

Main goal of this project is to utilize flow matching ideas to construct a model capable of performing compression (i.e., dimensionality reduction) while also learning and preserving intrinsic dynamics.

As of right now, this code only supports training networks on toy datasets and small image (toy-like) datasets.

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1 high-end NVIDIA GPU. All testing and development done using NVIDIA RTX 3090 and 4090 GPUs. Code tested for single-GPU training and generation, though it can easily be adapted for multi-GPU setting.
* 64-bit Python 3.8 and PyTorch 2.1.2. See https://pytorch.org for PyTorch install instructions.
* Python libraries: See [environment.yml](./environment.yml) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda env create -f environment.yml -n vfm`
  - `conda activate vfm`


## Training New Models

**Toy DataSets**

To train a model on a given toy dataset, run:

```.bash
torchrun --rdzv_endpoint=0.0.0.0:29501 toy_train_vfm.py --outdir=out --data_name=<dset_name> \
--data_dim=2 --dims_to_keep=2
```

Code supports following options for toy datasets:

1) 2D doublecircles (`doublecircles`)
2) 2D vanderpol (`vanderpol`)
3) 2D SDE (`doublesdeorbit`)
4) 3D rossler (`rossler`)
5) 3D Lorenz (`lorenz63`)
6) Higher-D Lorenz (`lorenz63`) - change param `d` to change dimensionality of Lorenz system.
7) Balls Image dataset (`balls`).

For additional details and argument options please check actual `toy_train_vfm.py` script.

## Simulating Evolution of Flow and/or Dynamics Networks

**Toy Datasets**

To simulate entire trajectories for several possible flow times (`taus`), use `toy_dynamics_traj_sim.py`, as follows:

```.bash
torchrun --rdzv_endpoint=0.0.0.0:29501 toy_dynamics_traj_sim.py --outdir=out --network=net.pkl \
--n_trajs=100 --traj_len=1000 --taus=0.0,0.25,0.50,0.75,1.0 --data_name=<dset_name> --data_dim=2 --dims_to_keep=2
```

This will simulate 100 full trajectories, each with 1K time steps, with flow time `tau` evaluated at each one of the float values
passed to `--taus`. Network here should be a trained net saved inside a pickled file. By default script saves trajectories as .npz files
inside a subdirectory called `dynamics_traj_net_sims` which is created under the given output root dir. Additionally, script will
also save a .json file containing all the parameters passed on/used to simulate the trajectories.

Another option is to alternate simulation/integration between the dynamics and flow networks. This generates trajectories which have
different pieces/number of steps evaluated at different (increasing) flow times `taus`. To run this option, use:

```.bash
torchrun --rdzv_endpoint=0.0.0.0:29501 toy_alternating_flow_dyn_traj_sim.py --outdir=out --network=net.pkl \
--tps_per_tau=200 --num_taus=5 --data_name=<dset_name> --data_dim=2 --dims_to_keep=2 --n_trajs=100
```

This will construct 100 trajectories using alternating integration between dynamics and flow networks. Number of time steps to evolve
in dynamics net before changing flow time `tau` is indicated by `--tps_per_tau` and the total number of linearly spaced transitions/increases
in flow time `tau` is given by `--num_taus`. Code will always start from `tau==0` and end in `tau==1`. Note that total trajectory length is
determined by both `--tps_per_tau` and `--num_taus`.

## Branch info/basics

Most of the info you care about will be in the following 3 branches:

`main` -- Contains original code with implementation of simple reconstruction-based pretraining loss (a.k.a. encoder loss) and using torch autograd functionality to compute Lie derivative loss ONLY AFTER a given pre-training number of Mimgs. We turned away from this due to how slow/costly it was to actually compute desired Jacobian vector product (JVP) using torch atugrad functionality.

`global_new_param_new_pt` -- Updates to `main` to use instead conditional Lie derivative form we mentioned in notes (as opposed to autograd computed Lie derivative/JVP). Here parameterization of L and D is global, and only encoder/proposal means are locally parameterized.

`local_new_param_new_pt` -- Updates to `main` to use instead conditional Lie derivative form we mentioned in notes (as opposed to autograd computed Lie derivative/JVP). Here L, D, and means are ALL locally parameterized.

Of note, both `global_new_param_new_pt` and `local_new_param_new_pt` begin pre-training with flow net loss, dyn net loss, and simple encoder reconstruction loss. After specified pre-training number of Mimgs, conditional Lie loss component is added to global loss. This set up seems to be more stable from toy experiments I ran back in April/May.

You will also see:

`JVP-Investigation` -- Contains some work an undergrad RA working with us on Spring did trying to optimize autograd computation of our desired JVP. This is obsolete at this point, so DO NOT WORRY ABOUT IT.

`model_comparison` -- Branch containing some of the original notebooks Miles Martinez created to run models we will likely want to compare our vfm model against. Most of these nbs are now incorporated into `comparison_testing_notebooks` subdir under main so feel free to check these out. Miles will probably be the one working more on these for additional experiments. Overall, we tried to keep code for comparison models strictly on these nbs and separate from rest of repo (we might wish to integrate it in future - TBD).
