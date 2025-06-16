## Installation

```
## Setup the environment
git clone https://github.com/Daeen/Seg4DGaussian.git
cd Seg4DGaussian
git submodule update --init --recursive
conda create -n SADG python=3.8 -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install opencv-python plyfile tqdm scipy wandb opencv-python scikit-learn lpips imageio[ffmpeg] dearpygui kmeans_pytorch hdbscan scikit-image bitarray
python -m pip install submodules/diff-gaussian-rasterization
python -m pip install submodules/simple-knn

## Install SAM weights
cd dependency
bash install.bash

## For enabling Text Prompt
git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
```

Note: If you have an error from Grounding-DINO: `TypeError: annotate() got an unexpected keyword argument 'labels'`, install `Supervision` to the 0.21.0 version

```
pip install supervision==0.21.0
```

## Dataset Preparation

See [here](./docs/prepare_dataset.md)

## Train

~~See [here](./mask_script.sh)

## Render

See [here](./docs/render.md)

## Evaluation on our Mask-Benchmarks

See [here](./docs/evaluation.md)

## BibTex and

```
@article{li2024sadg,
    title={SADG: Segment Any Dynamic Gaussian Without Object Trackers},
    author={Li, Yun-Jin and Gladkova, Mariia and Xia, Yan and Cremers, Daniel},
    journal={arXiv preprint arXiv:2411.19290},
    year={2024}
}
```
