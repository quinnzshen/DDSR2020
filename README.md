# Dense Depth Summer Research (DDSR2020)
In this repository, we replicate the research of [Monodepth2 - Digging into Self-Supervised Monocular Depth Prediction (ICCV 2019)](https://arxiv.org/abs/1806.01260). Additionally, we explore a few extensions to this research and find that replacing the original residual neural network (ResNet) encoder with a densely connected convoluational network (DenseNet) encoder results in better metrics and faster model covergence. We find that for similar levels of accuracy in dense depth - the DenseNet architecture is more efficient in learned parameters and computation at the trade-off of memory usage during training.

<p align="center">
  <img align="center" src="assets/densenet_ms.gif" alt="Qualitative DenseNet121(MS) results on KITTI dataset scene." width="700" /><br>
  <i>An example scene from the KITTI dataset.</i><br>
  <i>Top: Original input image, Middle: Baseline Monodepth2, Bottom: DenseNet121 Result.</i>
</p>

## Environment setup
1. [Install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) Anaconda. You can install Miniconda if space is limited.
2. Create your `ddsr` anaconda environment where you will be doing development. In terminal:
```
$ conda env create -f ddsr_environment.yml
```
3. Activate `ddsr` anaconda environment:
```
$ conda activate ddsr
```
4. Create a Jupyter kernel for your `ddsr` anaconda environment.
```
$ python -m ipykernel install --user --name ddsr --display-name "Python (ddsr)"
```

## Testing your environment
1. Launch jupyter and ensure you can run the import statements.
```
$ jupyter notebook
```

## Introducing new packages and dependencies into ddsr_environment.yml
1. While on your ddsr environment
```
conda env export --no-builds > ddsr_environment.yml
```

## Contributors
- Alex Jiang
- Aaron Marmolejos
- Kevin Z Shen
- Quinn Z Shen
- Evan Wang
