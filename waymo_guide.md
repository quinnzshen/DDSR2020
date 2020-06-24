# Waymo Guide

Quick tutorial on how to use the Waymo dataset

## Downloading data
1. Visit https://waymo.com/open/download/
2. Install training_0000.tar
3. Extract using winzip

You should have 24 TFrecord files in total.

## Installing Waymo-Open-Dataset
1. Upgrade pip with command "python -m pip install --upgrade pip" or "pip3 install --upgrade pip"
2. Try the command "pip3 install waymo-open-dataset-tf-2-1-0==1.2.0" or "pip install waymo-open-dataset"
3. If that doesn't work, clone the repo here: https://github.com/waymo-research/waymo-open-dataset, and add it to your python path with sys.path.append(directory)

Note: You might need to switch to python 3.6, not sure if 3.7 or 3.8 are suported

## Packages to add
1. Tensorflow, version = 2.1.0. 
2. Sys (if pip install waymo-open-dataset doesn't work)
3. Waymo_open_dataset (referenced above)

## Running my code
1. Edit the waymo_loader_test_config.yml file to match your setup

## Questions
Contact me @ alexjiang8715@gmail.com or via Slack