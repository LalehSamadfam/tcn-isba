#!/usr/bin/python2.7
from itertools import groupby

import torch
import numpy as np
import random
from cv2 import resize
import cv2


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='float32')[y]


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, segmentation_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.segmentation_path = segmentation_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.class_weights = torch.empty(num_classes)
        self.transcripts = dict()
        self.occurance = []
        self.lens = dict()

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def make_transcripts(self):
        for vid in self.list_of_examples:
            file_ptr = open(self.segmentation_path + vid, 'r')
            seg_content = file_ptr.readlines()
            size = np.repeat(0, int(seg_content[-1].split()[0].split('-')[1]))
            gt = np.ndarray(size.shape, dtype='object')  # segmentation ground truth
            for act in seg_content:
                time, label = act.split()
                gt[(int(time.split('-')[0]) - 1):int(time.split('-')[1])] = label
            transcript = [i[0] for i in groupby(gt)]  # sequence of actions in label
            pseudo_gt = np.zeros(len(transcript), dtype='uint8')  # sequence of actions in index
            for i in range(len(pseudo_gt)):
                pseudo_gt[i] = int(self.actions_dict[transcript[i]])
            self.transcripts[vid] = pseudo_gt
            self.lens[vid] = size.shape[0]
        file_ptr.close()

    def make_occurance(self):
        for vid in self.list_of_examples:
            file_ptr = open(self.segmentation_path + vid, 'r')
            seg_content = file_ptr.readlines()
            size = np.repeat(0, int(seg_content[-1].split()[0].split('-')[1]))
            gt = np.ndarray(size.shape, dtype='uint8')  # segmentation ground truth
            for act in seg_content:
                time, label = act.split()
                gt[(int(time.split('-')[0]) - 1):int(time.split('-')[1])] = self.actions_dict[label]
            self.occurance.append(np.unique(gt))
        file_ptr.close()

    def set_class_weights(self):
        classes = np.ones(self.num_classes) * 0.000001
        for vid in self.list_of_examples:
            file_ptr2 = open(self.segmentation_path + vid, 'r')
            seg_content = file_ptr2.readlines()
            for act in seg_content:
                time, label = act.split()
                action_class = self.actions_dict[label]
                classes[action_class] += 1
        classes = (1 / classes) ** 0.5
        classes /= (1 / 48 * np.sum(classes))
        print(classes)
        self.class_weights = torch.tensor(classes)


    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]  # list of files
        file_ptr.close()
        print("reading data names finished!")
        # random.shuffle(self.list_of_examples) #
        #self.set_class_weights()  # class weights
        #print("calss weights are all set!")
        self.make_transcripts()
        print("transcripts are all created!")
        self.make_occurance()
        print("occurance arrays are all created!")


    def loss(self, predictions):
        ocurance_categorical = np.array([np.sum(to_categorical(y, self.num_classes), 0)
                                         for y in self.occurance])
        class_prob = np.array([np.max(p, 0) for p in predictions])
        class_prob = class_prob.clip(min=0)
        temp = np.sum(-(ocurance_categorical * np.log(class_prob + 1e-7)
                        + (1 - ocurance_categorical) * np.log(1 - class_prob + 1e-7)), axis=1)
        loss = np.mean(temp)
        return loss

    def generate_target(self, predictions):
        rho = 0.02
        change_stack = []
        batch = self.list_of_examples
        for index, prediction in enumerate(predictions):
            transcript = np.array(self.transcripts[batch[index]])
            for i in range(0, transcript.shape[0] - 1):
                t = int(prediction.shape[0] / transcript.shape[0]) * (i + 1)
                if transcript[i] != transcript[i + 1]:
                    if prediction[t][transcript[i]] - prediction[t][transcript[i + 1]] > rho:
                        change_stack.append([i + 1, transcript[i]])
                    elif prediction[t][transcript[i + 1]] - prediction[t][transcript[i]] > rho:
                        change_stack.append([i + 1, transcript[i + 1]])
            if len(change_stack) > 0:
                for i in range(len(change_stack)):
                    transcript = np.insert(transcript, change_stack[i][0] + i, change_stack[i][1])
            change_stack = []
            self.transcripts[batch[index]] = transcript

    def to_target(self, sequence, len):
        seq = to_categorical(sequence, self.num_classes)
        dim = (self.num_classes, len)
        pseudo_gt_expanded = resize(seq, dim, interpolation=cv2.INTER_LINEAR)
        return np.array(pseudo_gt_expanded)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]  # TODO change if you change batch size
        self.index += batch_size

        batch_input = []
        batch_target = []
        length_of_sequences = []
        for vid in batch:
            file_ptr2 = self.features_path + vid
            features = np.loadtxt(file_ptr2).T
            #features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            transcript = self.transcripts[vid]
            vid_len = min(self.lens[vid], features.shape[1])
            pseudo_gt = self.to_target(transcript, vid_len)
            batch_input.append(features[:, ::self.sample_rate])
            new = pseudo_gt[::self.sample_rate]
            batch_target.append(new)
            length_of_sequences.append(len(new))

        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0],
                                         max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), self.num_classes, max(length_of_sequences),
                                         dtype=torch.float) * (-100)

        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i].T)
        return batch_input_tensor, batch_target_tensor
