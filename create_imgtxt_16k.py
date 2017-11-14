import os

SCRIPT_DIR = os.getcwd()
TRAINING_FOLDER = 'Input_spectrogram_16k/Training/'
talkers = os.listdir(TRAINING_FOLDER)
talkers.sort()

spec_dict_train = {}

for l in talkers:
    spec_dict_train[l] = sorted(os.listdir(TRAINING_FOLDER + l))

txtfile = open('./img_set_16k_train.txt', mode='w')

for l in talkers:
    os.chdir(TRAINING_FOLDER + str(l))
    for i in range(0, len(spec_dict_train[l])):
        txtfile.write(str(spec_dict_train[l][i]) + " " + str(l))
        txtfile.write("\n")
    os.chdir(SCRIPT_DIR)

txtfile.close()


VALIDATION_FOLDER = 'Input_spectrogram_16k/Validation/'
talkers = os.listdir(VALIDATION_FOLDER)
talkers.sort()

spec_dict_val = {}

for l in talkers:
    spec_dict_val[l] = sorted(os.listdir(VALIDATION_FOLDER + l))

txtfile = open('./img_set_16k_val.txt', mode='w')

for l in talkers:
    os.chdir(VALIDATION_FOLDER + str(l))
    for i in range(0, len(spec_dict_val[l])):
        txtfile.write(str(spec_dict_val[l][i]) + " " + str(l))
        txtfile.write("\n")
    os.chdir(SCRIPT_DIR)

txtfile.close()
