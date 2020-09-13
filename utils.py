import cv2
import os
import numpy as np
import glob
from tqdm import tqdm
import batch_gen


def generate_target(transcript_path, temp_results_dir, vid_list_file):
    transcript = []
    prediction = []
    rho = 0.5
    change_stack =[]
    for i in range(0, transcript.shape[0] - 1):
        t = prediction.shape[0]/transcript.shape[0] * i + 2
        if (transcript[i] != transcript[i + 1]):
            if prediction[t][transcript[i]] - prediction[t][transcript[i + 1]] > rho:
                #transcript = np.insert(transcript, t + 1, )
                change_stack.insert([t + 1, transcript[i]])
            elif prediction[t][transcript[i + 1]] - prediction[t][transcript[i]] > rho:
                change_stack.insert([t + 1, transcript[i + 1]])

    for i in len(change_stack):
        np.insert(transcript, change_stack[i][0] + i, change_stack[i][1])

    #TODO read transcript and prediction, and save new transcripts at the end

