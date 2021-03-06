#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import time


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes))
                                     for s in range(num_stages-1)])

    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.relu(out))
            out = s(out)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
                                     for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)


    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        #out = (F.relu(out))
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, weights):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.num_classes = num_classes
        self.weights = weights

    def no_grad_uniform(self, tensor, a, b):
        with torch.no_grad():
            return tensor.uniform(a, b)

    def weight_init(self, m):
        offset = 0.05
        if type(m) == nn.Conv1d:
            m.weight = self.no_grad_uniform(m.weight, 1 - offset, 1 + offset)
            m.bias.data.fill_(0.01)

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device, isba_loop):
        self.weight_init(self.model)
        self.model.train()
        self.model.to(device)
        #self.weights.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        print("lets start training!")

        start = time.time()
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            while batch_gen.has_next():
                batch_input, batch_target, vid = batch_gen.next_batch(batch_size)
                batch_input, batch_target = batch_input.to(device), batch_target.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)

                loss = 0
                for p in predictions:
                    bce = self.bce(p.squeeze(), batch_target.squeeze().type_as(p))
                    #a = np.sum(self.weights)
                    #bce_weighted = torch.mul(self.weights, bce.transpose(0, 1))
                    #bce = bce_weighted.mean()
                    loss += bce
                epoch_loss += loss
                loss.backward()
                #print(self.model.stage1.conv_1x1.weight.grad)
                optimizer.step()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/" + str(isba_loop) + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/" + str(isba_loop) + "/epoch-" + str(epoch + 1) + ".opt")
            end = time.time()
            print("epoch=", epoch, ", loss=", "{:.4f}".format(epoch_loss / len(batch_gen.list_of_examples)), ", time:", "{:.2f}".format(end - start), "s")

            #print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),float(correct)))

    def predict(self, model_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        temp_res = []
        temp_res_categorical = []
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                file_ptr2 = features_path + vid
                features = np.loadtxt(file_ptr2).T
                #features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                probs = predictions[-1].data.squeeze().cpu()
                probs = F.softmax(probs, dim = 1)
                probs = probs.numpy().T
                _, predicted = torch.max(predictions[-1].data, 1)
                temp_res_categorical.append(probs)
        print("min: ", np.min(temp_res_categorical[0]), "max: ", np.max(temp_res_categorical[0]) )
        return temp_res_categorical


    def predict_test(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        temp_res = []
        temp_res_categorical = []
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                file_ptr2 = features_path + vid
                features = np.loadtxt(file_ptr2).T
                #features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                probs = predictions[-1].data.squeeze().cpu().numpy().T
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    t = predicted[i].item()
                    for key, value in actions_dict.items():
                        if value == t:
                            label = key
                    recognition = np.concatenate((recognition, [label]*sample_rate))
                temp_res.append(recognition)
                temp_res_categorical.append(probs)
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
        print("min: ", np.min(temp_res_categorical[0]), "max: ", np.max(temp_res_categorical[0]) )
        return temp_res, temp_res_categorical
