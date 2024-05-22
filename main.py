
import hydra
import os, sys

sys.path.append("..")

try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    print("Graph tool could not be loaded")

import torch

from bridge.runners.ipf import IPFSequential


# SETTING PARAMETERS


@hydra.main(config_path="./conf", config_name="config")
def main(args):

    print("Directory: " + os.getcwd())
    ipf = IPFSequential(args)
    if not args.test:
        ipf.train()
    else:
        ipf.test()


if __name__ == "__main__":
    main()
