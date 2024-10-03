import torch
from urllib.request import urlopen
from PIL import Image
import timm

timm.create_model('resnest50d.in1k')