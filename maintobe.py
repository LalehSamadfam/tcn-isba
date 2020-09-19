#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import utils
import os
import argparse
import random
import numpy as np


# init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="breakfast")
parser.add_argument('--split', default='2')

args = parser.parse_args()

num_stages = 1 #TODO change it to 2
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1
lr = 0.0005
num_epochs = 1 #TODO change it to 50

# use the full temporal resolution @ 15fps
sample_rate = 1

vid_list_file = "./data/" + args.dataset + "/splits/train.split" + args.split + ".bundle"
vid_list_file_tst = "./data/" + args.dataset + "/splits/test.split" + args.split + ".bundle"
features_path = "./data/" + args.dataset + "/features/"
#gt_path = "./data/" + args.dataset + "/groundTruth/"
segmentation_path = "./data/" + args.dataset + "/segmentation/"

mapping_file = "./data/" + args.dataset + "/mapping.txt"

model_dir = "./models/" + args.dataset + "/split_" + args.split
results_dir = "./results/" + args.dataset + "/split_" + args.split
temp_results_dir = "./temp_results/" + args.dataset + "/split_" + args.split

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)


# train

no_change = 1
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, segmentation_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    weights = batch_gen.set_class_weights()
    trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes, weights)
    while(no_change):
        trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)
        trainer.predict(model_dir, temp_results_dir, features_path, vid_list_file, num_epochs, actions_dict, device,
                    sample_rate)
        utils.generate_target(segmentation_path, temp_results_dir, vid_list_file)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device,
                    sample_rate)
