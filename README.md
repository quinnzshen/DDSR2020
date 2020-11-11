# DDSR2020
Dense Depth Summer Research 2020 repository.

<p align="center">
  <img align="center" src="assets/example_asset.gif" alt="This is an example gif for DDSR2020." width="700" /><br>
  <i>This is an example visual that we'll add for DDSR2020.</i>
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
