import logging
import torch
import sys
import os
import random
import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import hiddenlayer as h
from torchviz import make_dot

if __name__ == "__main__":

    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        default=parentdir + "/configs/extra_4.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path[0] = parentdir

    from solver.ml_core.trainer import Trainer
    from solver.ml_core.datasets import GraphDataset
    from networks.pseudo_tilingnn import PseudoTilinGNN as Gnn

    Path(config["training"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(config["training"]["log_dir"], "output.log"),
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
        level=getattr(logging, config["training"]["logging_level"]),
        datefmt='%Y-%m-%d %H:%M:%S')

    torch.manual_seed(config["training"]["seed"])
    random.seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn = Gnn(network_depth=config["network"]["depth"],
              network_width=config["network"]["width"],
              output_dim=config["network"]["output_dim"]).to(device)

    print(gnn)
    
    # sampledata = torch.ones(2,1)
    # e_idx = torch.tensor([[   0,    1,],[   1,    0,   ]])
    # out = gnn(sampledata, e_idx)
    # print(out)

    # with SummaryWriter("./vislog", comment="sample_model_visualization") as sw:
    #     sw.add_graph(gnn, (sampledata, e_idx))

    # torch.save(gnn, "./vislog/modelviz.pt")

    # out = gnn(sampledata, e_idx)
    # g = make_dot(out)
    # g.render('gnn', view=True)

    # vis_graph = h.build_graph(gnn, (sampledata, e_idx))
    # vis_graph.theme = h.graph.THEMES["blue"].copy()
    # vis_graph.save("./demo1.png") 

   
