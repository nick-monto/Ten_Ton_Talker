import os
import numpy as np
import fnmatch
import pandas as pd

def returnNotMatches(a, b):
    return [[x for x in a if x not in b], [x for x in b if x not in a]]


df_train = pd.read_table('img_set_16k_train.txt',
                         delim_whitespace=True,
                         names=['stimulus', 'language'])

stim_train = df_train['stimulus']
labels_train = pd.get_dummies(df_train['language'])
labels_train = labels_train.values


df_val = pd.read_table('img_set_16k_val.txt',
                       delim_whitespace=True,
                       names=['stimulus', 'language'])

stim_val = df_val['stimulus']
labels_val = pd.get_dummies(df_val['language'])
labels_val = labels_val.values

stim_train = pd.Series(data=stim_train)

unique_train = stim_train.str.extractall('(?<=)(.*)(_0)+')[[0]].values
unique_val = stim_val.str.extractall('(?<=)(.*)(_0)+')[[0]].values

used_set = np.concatenate((unique_train, unique_val))
compare_set = []

for i in range(0, len(used_set)):
    compare_set.append(used_set[i][0] + ".wav")

compare_set.sort()

SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = 'Input_audio_wav_16k/'
talkers = os.listdir(INPUT_FOLDER)
talkers.sort()

audio_dict = {}

for l in talkers:
    audio_dict[l] = sorted(os.listdir(INPUT_FOLDER + l))

partial_list = []
for x in audio_dict.values():
    partial_list.append(x)

full_list = []
for x in range(0, len(partial_list)):
    for y in range(0, len(partial_list[x])):
        full_list.append(partial_list[x][y])

full_list.sort()

test_list = returnNotMatches(full_list, compare_set)

len(test_list[0])

txtfile = open('./test_set_16k.txt', mode='w')

for x in range(0, len(test_list)):
    for y in range(0, len(test_list[x])):
        txtfile.write(str(test_list[x][y]) + "\n")

txtfile.close()
