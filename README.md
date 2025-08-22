# High-numerical-aperture confocal volumetric mesoscope reveals collective mesoscale immune responses

This is the repository of parallel reconstruction pipeline for the paper "High-numerical-aperture confocal volumetric mesoscope reveals collective mesoscale immune responses".

## System requirements

We recommend using the following system requirements:
* 128 GB RAM or more
* NVIDIA RTX 3090 with 24 GB of VRAM or more
* Ubuntu 22.04 LTS
* Python 3.12

## Environment
There is no restriction on operation systems or specific python versions (as long as they are supported by Pytorch). 

This repository is tested on Python 3.12 and Pytorch 2.5.1.

Multiple GPUs are supported and recommended for optimal computational efficiency. 8 NVIDIA RTX 3090 GPUs on Ubuntu 22.04 LTS were tested.

## Installation

We strongly recommend to install the Pytorch 2.5.1 with CUDA support according to the Pytorch official website before installing the other dependencies.

We have provided the requirements file in the root directory of this repository.

You can install the necessary dependencies by running the following command:
```bash
pip install -r requirements.txt
```

## Optional TensorRT acceleration

Tensor-RT is supported to further accelerate the pipeline. To enable TensorRT acceleration, you will need to install TensorRT and torch2trt package. Then you have to recompile engine for the reconstruction model to match your computational platform.

You may check the [TensorRT official website](https://developer.nvidia.com/tensorrt) for more information about TensorRT.

For recompiling reconstruction model, please check the 'torch_2_onnx.py' file inside 'Code' directory.

Finally, you can enable TensorRT acceleration by setting the `use_tensorrt` option in the configuration file to `True`.

## Demo
### Download demo dataset

We have prepared a demo dataset for you to try out the pipeline. The dataset is available at [Zenodo](https://zenodo.org/records/16919909). Put the dataset at the "data" directory of this repository to run the demo with default configuration.

### Download the necessary dependencies

The necessary dependencies are available at [Zenodo](https://zenodo.org/records/16919833). Put the dependencies at the "reconstruction" directory of this repository to run the demo with default configuration.

### Run demo

Run the demo with default configuration:
```bash
python Code/recon_torch.py
```

This demo will run the pipeline on the demo dataset with 'RCConfig.json' at the root directory. The reconstructed volume will be saved at "result" folder. You can also specify the configuration file by adding the `--config` option.
