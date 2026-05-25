# Dynamic Compression Flows for Neuroscience Data

This repository contains code for **Dynamic Compression Flows (DCF)**, a flow-matching framework for learning low-dimensional representations of high-dimensional dynamical data while preserving temporal structure.

DCF learns two coupled vector fields:

1. A **compressive/generative flow** that maps between data space and a compressed latent representation.
2. A **dynamical flow** that models time evolution at each compression level.


## Repository structure

```text
.
├── dnnlib/                 # General utilities for data handling, model helpers, VAE helpers, and reusable infrastructure
├── notebook_analysis/      # Jupyter notebooks and analysis scripts used to reproduce paper figures and experiment analyses
├── plotting/               # Standalone plotting scripts
├── torch_cfm/              # Conditional flow matching and mini-batch OT utilities adapted from TorchCFM, see third-party references below
├── torch_utils/            # PyTorch utilities for distributed training, checkpointing, logging, and persistence
├── training/               # Core DCF training code, including losses, network architectures, and training loops
│
├── vfm_train_v7.py         # Main command-line training entry point for DCF experiments
├── run_example.txt         # Example commands for running representative experiments
├── environment.yml         # Conda environment specification
├── README.md               
└── .gitignore
```

The main training entry point is `vfm_train_v7.py`. Core model code lives in `training/`, paper analysis notebooks live in `notebook_analysis/`, and reusable plotting scripts live in `plotting/`.

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml -n vfm
conda activate vfm
```

## Data format

The main training script expects a data folder passed through `--data_path`.

The required file is:

```text
dataset_samples.npz
```

It must contain the key:

```text
samples
```

Supported sample formats include vector-valued trajectories,

```text
n_trials x T x D
```

and image-like trajectories,

```text
n_trials x T x C x H x W
```

Optional covariate files are:

```text
cov_dynamic_samples.npy
cov_static_samples.npy
```

Use the corresponding flags when covariates are available:

```bash
--use_dynamic_covariates
--use_static_covariates
```

Lag-history inputs for the dynamical flow are controlled by:

```bash
--lag_k <K>
```

## Quick start

A minimal training command has the following form:

```bash
python vfm_train_v7.py \
  --data_path data/<dataset_folder> \
  --data_name <dataset_name> \
  --outdir out/<experiment_name> \
  --k_max <max_latent_budget> \
  --k_target <effective_latent_dim, i.e., K ~ Geom(1/k_target)> \
  --lag_k <history_length> \
  --dt <time_step> \
  --duration <training_duration> \
  --batch <batch_size> \
  --batch_gpu <per_gpu_batch_size> \
  --dyn_arch <dynamics_network> \
  --flow_arch <compression_network> \
  --encoder_arch <encoder_network> \
  --alpha 1.0 \
  --beta 1.0 \
  --eta 1.0
```

See `run_example.txt` for complete commands for the rotating-ball and maze/neural examples.

## Example architectures

For vector-valued neural data, the paper uses MLP-based encoder, compression flow, and dynamical flow networks. A typical setup is:

```bash
--dyn_arch ToyMLP \
--flow_arch ToyMLP \
--encoder_arch Latent_MLP_VAE \
--encoder_depth 4 \
--encoder_width 256 \
--mlp_depth_cmp 4 \
--mlp_width_cmp 256 \
--mlp_depth_dyn 4 \
--mlp_width_dyn 512
```

For image-like data, the paper uses convolutional encoder-decoder networks:

```bash
--dyn_arch ToyConvUNet \
--flow_arch ToyConvUNet \
--encoder_arch Latent_LargeCNN_VAE \
--conv_ch_cmp 32,64,128,256 \
--conv_embed_cmp 256 \
--conv_ch_dyn 32,64,128,256 \
--conv_embed_dyn 256 \
--conv_ch_enc 32,64,128,256
```

## Important options

### Latent dimension

```bash
--k_max <INT>
```

sets the maximum latent budget.

```bash
--k_target <INT>
```

fixes the effective nested-dropout latent dimension. If omitted, adaptive nested dropout is used.

### Loss weights

```bash
--alpha
```

weights the compressive flow-matching loss.

```bash
--beta
```

weights the dynamical flow-matching loss.

```bash
--eta
```

weights the encoder alignment loss.


## Outputs

Training outputs are written to:

```text
--outdir
```

The script also saves processed data under:

```text
<outdir>/data/
```

Typical outputs include processed trajectories, covariates, lag-history arrays, network snapshots, training state dumps, and logs.


## Third-party code and method references

Parts of this repository build on or adapt utilities from TorchCFM, including conditional flow matching and mini-batch optimal transport components. If you use the `torch_cfm/` components or the OT-CFM option, please also cite the original Conditional Flow Matching / OT-CFM work.

```bibtex
@article{tong2024improving,
  title = {Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport},
  author = {Tong, Alexander and Fatras, Kilian and Malkin, Nikolay and Huguet, Guillaume and Zhang, Yanlei and Rector-Brooks, Jarrid and Wolf, Guy and Bengio, Yoshua},
  journal = {Transactions on Machine Learning Research},
  year = {2024}
}
```

```bibtex
@software{torchcfm,
  title = {TorchCFM: Conditional Flow Matching},
  author = {Tong, Alexander and contributors},
  year = {2023},
  url = {https://github.com/atong01/conditional-flow-matching}
}
```


## Citation

If you use this code, please cite:

```bibtex
@inproceedings{wei2026dynamic,
  title = {Dynamic Compression Flows for Neuroscience Data},
  author = {Wei, Ganchao and de Albuquerque, Daniela and Martinez, Miles and Pan, Shiyang and Pearson, John},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  volume = {306},
  year = {2026}
}
```

Please update the BibTeX entry with final page numbers once available.

## License

Please add a repository-level license file before public release.