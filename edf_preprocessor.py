import numpy as np
import json
from py_linq import Enumerable
from pyedflib import highlevel
import progressbar
import urllib.request

class EDF_Preprocessor:
    
    def __init__(self, json_file_name):

        with open(json_file_name, 'r') as json_file:
            json_data = json.load(json_file)['Patients']
        self.__jenum = Enumerable(json_data)

    def __download(self, name, filename = 'temp.edf'):
        record = self.__jenum\
            .select(lambda it: it['Records'])\
            .select_many(lambda it: it)\
            .where(lambda it: it['RecordName'] == name)\
            .first_or_default()
        if record is None:
            raise ValueError('No such file')
        urllib.request.urlretrieve(record['Url'], filename)

    def __get_record(self, name, ignore_skip_status):
        record = self.__jenum\
            .select(lambda it: it['Records'])\
            .select_many(lambda it: it)\
            .where(lambda it: it['RecordName'] == name)\
            .first_or_default()
        if record is None:
            raise ValueError('No such file')
        skip_status = record['Skip']
        if not ignore_skip_status and skip_status:
            raise Exception('Bad file, must be skipped')
        return record

    def __get_records_range(self, offset, count):
        records = self.__jenum\
            .where(lambda it: not it['Skip'])\
            .select(lambda it: it['Records'])\
            .select_many(lambda it: it)\
            .where(lambda it: not it['Skip'])\
            .skip(offset)\
            .take(count)
        return records

    def get_data(self, name, ignore_skip_status = False, reshape=256):
        record = self.__get_record(name, ignore_skip_status = ignore_skip_status)
        
        self.__download(name)
        signals, _, _ = highlevel.read_edf('temp.edf')
        channels = record['Channels']
        signals = signals[channels]
        return signals.T.reshape(-1, reshape, 23)


    def get_data_range(self, offset, count, concat = True, reshape=256):
        records = self.__get_records_range(offset, count)
        data = list()
        
        bar = progressbar.ProgressBar(maxval = count).start()
        i = 0
        for record in records:
            name = record['RecordName']
            curdata = self.get_data(name, reshape=reshape)
            data.append(curdata)
            i += 1
            bar.update(i)
        bar.finish()
            
        if concat:
            return np.concatenate(data, axis=0)
        return data

    def get_labeled(self, name, ignore_skip_status = False, reshape=256, preictal_period=30):
        record = self.__get_record(name, ignore_skip_status = ignore_skip_status)
        data = self.get_data(name, ignore_skip_status = ignore_skip_status, reshape = reshape)

        seizures = record['Seizures']
        y = np.zeros(data.shape[0])
        last_end = 0

        preict = round(preictal_period * 60 * 256 / reshape)

        for seizure in seizures:
            start = round(seizure['Start'] / reshape)
            end = round(seizure['End'] / reshape)
            y[start : end] = 2

            preict_start = max(start - preict - 1, last_end)
            preict_end = start - 1
            y[preict_start : preict_end] = 1

            last_end = end

        return data, y

    def get_labeled_range(self, offset, count, concat = True, reshape=256, preictal_period=30):
        records = self.__get_records_range(offset, count)
        dataX = list()
        dataY = list()

        bar = progressbar.ProgressBar(maxval = count).start()
        i = 0
        for record in records:
            name = record['RecordName']
            X, y = self.get_labeled(name, reshape=reshape, preictal_period = preictal_period)
            dataX.append(X)
            dataY.append(y)
            i += 1
            bar.update(i)
        bar.finish()
            
        if concat:
            return np.concatenate(dataX, axis=0), np.concatenate(dataY, axis=0)
        return dataX, dataY