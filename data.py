import pandas as pd
import numpy as np
import csv


class PHMToolWearDataset(object):
    # extract all data
    cache_dir_path = '.cache/'
    total_signal_num = 315
    # only 1,4 and 6 is correct
    sample_label = [1, 4, 6]
    sample_loc = 1

    def __init__(self):
        self.force_update = False

    @property
    def res_data_path(self):
        return 'E:\\ubuntu_file\\rnn_in_prediction_tool_wear/c%s_wear.csv' % (self.sample_loc)

    @property
    def get_res_data_in_numpy(self):
        # remove cache because it's not needed
        res_csv_data = self.get_res_data_by_pandas
        res_array = np.array([np.array(i).reshape(3) for i in res_csv_data.values])
        # np.save(storage_path, res_array)
        return res_array

    @property
    def get_res_data_by_pandas(self):
        return pd.read_csv(self.res_data_path, index_col='cut')

    @property
    def res_data_storage(self):
        return 'phm_tool_wear_data'

    @property
    def get_tool_wear_data(self):
        storage_path = self.cache_dir_path + self.res_data_storage
        self.sample_loc = 1
        y_dat = self.get_res_data_in_numpy
        for i in [4, 6]:
            self.sample_loc = i
            y_dat = np.append(self.get_res_data_in_numpy, y_dat, axis=0)

        print(y_dat.shape)
        return y_dat




class RNNSeriesDataSet(object):

    def __init__(self,begin_timestep,end_timestep):
        a = PHMToolWearDataset()
        self.tool_wear_data = a.get_tool_wear_data
        self.max_tool_wear_data = np.max(self.tool_wear_data,axis=1)
        print(self.max_tool_wear_data.shape)
        self.begin_timestep = begin_timestep
        self.end_timestep = end_timestep

    def get_tool_wear_data(self):
        return self.tool_wear_data

    def get_individual_tool_wear_batches(self,tool_wear_data):
        round_number = tool_wear_data.shape[0]
        begin_series = []
        end_series = []
        for start_index in range(self.begin_timestep,round_number-self.end_timestep):
            x = tool_wear_data[start_index-self.begin_timestep:start_index]
            y = tool_wear_data[start_index:start_index+self.end_timestep]
            begin_series.append(x)
            end_series.append(y)
        return begin_series,end_series

    def get_rnn_data(self):
        x,y = [],[]
        for i in range(3):
            ix,iy = self.get_individual_tool_wear_batches(self.max_tool_wear_data[i*315:(i+1)*315])
            x.extend(ix)
            y.extend(iy)
            # print(len(ix))
        dat_x,dat_y =  np.array(x),np.array(y)
        return dat_x.reshape((dat_x.shape[0],dat_x.shape[1],1)),dat_y.reshape((dat_y.shape[0],dat_y.shape[1],1))

    def get_separate_rnn_data(self):
        x, y = [], []
        test_x,test_y = [],[]
        for i in range(3):
            ix, iy = self.get_individual_tool_wear_batches(self.max_tool_wear_data[i * 315:(i + 1) * 315])
            if i == 2:
                test_x.extend(ix)
                test_y.extend(iy)
            else:
                x.extend(ix)
                y.extend(iy)
                # print(len(ix))
        dat_x, dat_y = np.array(x), np.array(y)
        test_x,test_y = np.array(test_x),np.array(test_y)
        return dat_x.reshape((dat_x.shape[0], dat_x.shape[1], 1)),\
               dat_y.reshape((dat_y.shape[0], dat_y.shape[1], 1)),\
               test_x.reshape((test_x.shape[0],test_x.shape[1],1)),\
               test_y.reshape((test_y.shape[0],test_y.shape[1],1))

class CNNMonitoredDataSet(object):
    cnn_predict_wear = np.load("../.cache/Y_PRED.npy")
    cnn_max_predict_wear = cnn_predict_wear.max(axis=1)

    def __init__(self,begin_timestep,end_timestep):
        self.begin_timestep = begin_timestep
        self.end_timestep = end_timestep

    def get_individual_tool_wear_batches(self, tool_wear_data):
        round_number = tool_wear_data.shape[0]
        begin_series = []
        end_series = []
        for start_index in range(self.begin_timestep, round_number - self.end_timestep):
            x = tool_wear_data[start_index - self.begin_timestep:start_index]
            y = tool_wear_data[start_index:start_index + self.end_timestep]
            begin_series.append(x)
            end_series.append(y)
        return begin_series, end_series

    def get_rnn_data(self):
        x,y = [],[]
        for i in range(3):
            ix,iy = self.get_individual_tool_wear_batches(self.cnn_max_predict_wear[i*315:(i+1)*315])
            x.extend(ix)
            y.extend(iy)
            # print(len(ix))
        dat_x,dat_y =  np.array(x),np.array(y)
        return dat_x.reshape((dat_x.shape[0],dat_x.shape[1],1)),dat_y.reshape((dat_y.shape[0],dat_y.shape[1],1))



if __name__ == "__main__":
    tool_wear_data = PHMToolWearDataset()
    tool_wear_data = tool_wear_data.get_tool_wear_data
    np.save(".cache/Tool_wear",tool_wear_data)

    # a = RNNSeriesDataSet(2,5)
    # dat_x,dat_y = a.get_rnn_data()
    # print(dat_x.shape,dat_y.shape)