import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.serialization import load_lua

from PIL import Image
import argparse

import glob
import os

parser = argparse.ArgumentParser(description='Sketch simplification demo.')
parser.add_argument('--model', type=str, default='model_gan.t7', help='Model to use.')
parser.add_argument('--dir',   type=str, default='../data/car-design-edge/images/cars/',     help='Input image file.')
parser.add_argument('--out',   type=str, default='./simplified-car-design-edge/',      help='File to output.')
opt = parser.parse_args()

if not os.path.exists(opt.out):
    os.mkdir(opt.out)

use_cuda = torch.cuda.device_count() > 0

cache  = load_lua( opt.model )
model  = cache.model
immean = cache.mean
imstd  = cache.std
model.evaluate()

for f in glob.glob(opt.dir + '*.jpg'):

    data  = Image.open( f ).convert('L')
    w, h  = data.size[0], data.size[1]
    pw    = 8-(w%8) if w%8!=0 else 0
    ph    = 8-(h%8) if h%8!=0 else 0
    data  = ((transforms.ToTensor()(data)-immean)/imstd).unsqueeze(0)
    if pw!=0 or ph!=0:
        data = torch.nn.ReplicationPad2d( (0,pw,0,ph) )( data ).data

    if use_cuda:
        pred = model.cuda().forward( data.cuda() ).float()
    else:
        pred = model.forward( data )
    filename = f.split('/')[-1]
    save_image( pred[0], opt.out + filename )


