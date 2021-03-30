"""run_infer.py

Usage:
  run_infer.py [options] [--help]

  run_infer.py --version
  run_infer.py (-h | --help)
Options:
  -h --help                             Show this screen.
  --version                             Show version.
  --gpu=<id>                            GPU , only one gpu can be used. [default: 1]
  --input_path=<path>                   Input WSI folder(.svs file only)[default: ./dataset]
  --output_path=<path>                  output of th model[default: ./output]
  --step_size=<n>                       step_size for sliding window[default: 256]
  --show_samples=BOOL                   show compressed_sample[default: True]
"""
from docopt import docopt
from termcolor import cprint
import os
import torch
from infer import inferManager
if __name__ == '__main__':
    args = docopt(__doc__, help=False,options_first=True,version='PAIP2019 v.0.2')

    ##GPU
    gpu_list=args.pop("--gpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    nr_gpus = torch.cuda.device_count()
    cprint("Detect GPU: %d" % nr_gpus,"red")

    args = {k.replace('--', '') : v for k, v in args.items()}

    ##check args
    infer=inferManager.InferManager(args)
    infer.run()