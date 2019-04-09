import numpy as np
import pandas as pd
from ninaeval.utils.gt_tools import refine_start_end
import copy

def extract_myo_all_csv(path, loaded_nina, subject, exercise):
    """
        Fills the loaded_nina object with all data in a "myo_all_data.csv", obtained from our data collection GUI.

    :param path: Path to "myo_all_data.csv"
    :param loaded_nina: A dictionary that will be filled with the following structure:

                               {
                                   "s1": {
                                               "E1": {... },
                                               "E2": {... },
                                               "E3": {... }
                                           }

                                   "s2:" {
                                               "E1": {... }, ...
                                           }

                                   ...
                               }

    :param subject: Subject (e.g. "s1", "s2", ...)
    :param exercise: Exercise (e.g. "E1", "E2", ...)
    """

    # Convert CSV file to dictionary of lists
    df          = pd.read_csv(path, skipinitialspace=True)
    df_dict     = df.to_dict(orient='list')

    # Extract EMG data columns
    emg_1   = np.array(df_dict['D1_EMG_1'])
    emg_2   = np.array(df_dict["D1_EMG_2"])
    emg_3   = np.array(df_dict["D1_EMG_3"])
    emg_4   = np.array(df_dict["D1_EMG_4"])
    emg_5   = np.array(df_dict["D1_EMG_5"])
    emg_6   = np.array(df_dict["D1_EMG_6"])
    emg_7   = np.array(df_dict["D1_EMG_7"])
    emg_8   = np.array(df_dict["D1_EMG_8"])
    emg_9   = np.array(df_dict["D2_EMG_1"])
    emg_10  = np.array(df_dict["D2_EMG_2"])
    emg_11  = np.array(df_dict["D2_EMG_3"])
    emg_12  = np.array(df_dict["D2_EMG_4"])
    emg_13  = np.array(df_dict["D2_EMG_5"])
    emg_14  = np.array(df_dict["D2_EMG_6"])
    emg_15  = np.array(df_dict["D2_EMG_7"])
    emg_16  = np.array(df_dict["D2_EMG_8"])

    # Extract magnetometer data columns
    mag_1 = np.array(df_dict["D1_OR_W"])
    mag_2 = np.array(df_dict["D1_OR_X"])
    mag_3 = np.array(df_dict["D1_OR_Y"])
    mag_4 = np.array(df_dict["D1_OR_Z"])
    mag_5 = np.array(df_dict["D2_OR_W"])
    mag_6 = np.array(df_dict["D2_OR_X"])
    mag_7 = np.array(df_dict["D2_OR_Y"])
    mag_8 = np.array(df_dict["D2_OR_Z"])

    # Extract accelerometer data columns
    acc_1 = np.array(df_dict["D1_ACC_1"])
    acc_2 = np.array(df_dict["D1_ACC_2"])
    acc_3 = np.array(df_dict["D1_ACC_3"])
    acc_4 = np.array(df_dict["D2_ACC_1"])
    acc_5 = np.array(df_dict["D2_ACC_2"])
    acc_6 = np.array(df_dict["D2_ACC_3"])

    # Extract gyroscope data columns
    gyro_1 = np.array(df_dict["D1_GYRO_1"])
    gyro_2 = np.array(df_dict["D1_GYRO_2"])
    gyro_3 = np.array(df_dict["D1_GYRO_3"])
    gyro_4 = np.array(df_dict["D2_GYRO_1"])
    gyro_5 = np.array(df_dict["D2_GYRO_2"])
    gyro_6 = np.array(df_dict["D2_GYRO_3"])

    #
    # Stack numpy arrays to create 2D data (sample number, channel), per data type
    #
    emg_data    = np.stack((emg_1, emg_2, emg_3, emg_4, emg_5, emg_6, emg_7, emg_8, emg_9, emg_10, emg_11, emg_12,
                                emg_13, emg_14, emg_15, emg_16), axis=1)
    acc_data    = np.stack((acc_1, acc_2, acc_3, acc_4, acc_5, acc_6), axis=1)
    gyro_data   = np.stack((gyro_1, gyro_2, gyro_3, gyro_4, gyro_5, gyro_6), axis=1)
    mag_data    = np.stack((mag_1, mag_2, mag_3, mag_4, mag_5, mag_6, mag_7, mag_8), axis=1)


    #
    # Correct labels assigned
    #
    new_labels = copy.deepcopy(df_dict["Label"])

    for i in range(len(new_labels)):
        if new_labels[i] == -1:
            new_labels[i] = 0

    for j in range(len(new_labels)):
        new_labels[j] = [new_labels[j]]

    repetitions = [None for x in range(len(new_labels))]


    ####################################################################################################################
    ####################################################################################################################
    #
    # Refine original ground truth labels, using ground truth refinement algorithm (only using EMG data)
    #
    ####################################################################################################################
    ####################################################################################################################
    cur_idx = 0

    while cur_idx < len(new_labels):
        cur_label = new_labels[cur_idx]

        if (cur_label == [-1]) or (cur_label == [0]):
            cur_idx += 1
        else:
            start_idx   = cur_idx

            while (cur_idx < len(new_labels)) and (new_labels[cur_idx] == cur_label):
                cur_idx += 1
            end_idx = cur_idx - 1

            best_start, best_end = refine_start_end(emg_data, start_idx, end_idx)

            if (best_start is not None) and (best_end is not None):

                # For debugging purposes
                #
                # import matplotlib.pyplot as plt
                # plt.plot(emg_data[start_idx: end_idx])
                # plt.show()
                # plt.plot(emg_data[best_start: best_end])
                # plt.show()

                for i in range(start_idx, end_idx+1):
                    new_labels[i] = [0]
                for j in range(best_start, best_end):
                    new_labels[j] = cur_label
            else:
                #print("Unable to refine {}.".format((exercise, cur_idx)))
                pass

            cur_idx = end_idx + 400



    ####################################################################################################################
    ####################################################################################################################
    #
    # Update repetition number
    #
    ####################################################################################################################
    ####################################################################################################################
    cur_rep     = 1
    cur_idx     = 0
    inital_lab  = None
    started_rep = False


    while cur_idx < len(new_labels):
        cur_label = new_labels[cur_idx]

        if (cur_label == [-1]) or (cur_label == [0]):

            if started_rep:
                started_rep  = False
                cur_rep     += 1

        else:
            if not started_rep:
                inital_lab  = cur_label
                started_rep = True

            if cur_label != inital_lab:
                cur_rep = 1

        repetitions[cur_idx] = cur_rep
        cur_idx             += 1


    ####################################################################################################################
    ####################################################################################################################
    #
    # Store new data
    #
    ####################################################################################################################
    ####################################################################################################################
    if not subject in loaded_nina:
        loaded_nina[subject] = {}
    if not exercise in loaded_nina[subject]:
        loaded_nina[subject][exercise] = {}

    loaded_nina[subject][exercise]["emg"]           = emg_data
    loaded_nina[subject][exercise]["acc"]           = acc_data
    loaded_nina[subject][exercise]["gyro"]          = gyro_data
    loaded_nina[subject][exercise]["mag"]           = mag_data
    loaded_nina[subject][exercise]["restimulus"]    = new_labels
    loaded_nina[subject][exercise]["rerepetition"]  = repetitions