import numpy as np
import os
import torch
import argparse
import random, string,json
import matplotlib.pyplot as plt

from PIL import Image
from copy import deepcopy
from torch.nn import functional as F

from models.generators import *
from models.encoders import *
from models.configurations import configurations


def main():
    """
    Main function that parses arguments
    """
    parser = argparse.ArgumentParser(description="Calculate metrics given a path of saved generator")
    parser.add_argument("-f", "--generator", default="./../results/")