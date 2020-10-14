import glob
import os
import time

try:
    timer = time.time_ns
    time_unit = "ns"
    res_fac=10**9
except:
    timer = time.time
    time_unit = "s"
    res_fac=1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataRecorder():
    def __init__(self, folder="", basename=f"datarecorder"):

        self._folder = folder
        self._basename = basename
        self.reset()

        self._backup_min_time = 30
        self._backup_max_time = 120
        self._backup_min_data = 1
        self._backup_max_data = 200
        self._backupcount = 2
        self._save_rules_max_lines = 10 ** 4

    def label_to_col_index(self, col_array):
        col_indexes = col_array.copy()
        for i, j in enumerate(col_indexes):
            if isinstance(j, int):
                assert j < len(self._labels), f"y: {j} not in the range of labels"
            else:
                assert j in self._labels, f"y: {j} is not in labels"
                col_indexes[i] = self._labels.index(j)
        return col_indexes

    def data_point(self, **kwargs):
        self._current_index += 1
        newlabel = False
        for key in kwargs.keys():
            if key not in self._labels:
                newlabel = True
                self._labels.append(key)
                self._data.append([np.nan] * self._current_index)

        nd = [kwargs.get(l, np.nan) for l in self._labels]
        for i, d in enumerate(nd):
            self._data[i].append(d)

        if newlabel:
            self._sort_columns()
        self._check_backup()
        self._check_save()

    def sort(self, index, inplace=True):
        np_data = self.as_array()
        index = self.label_to_col_index([index])[0]
        indexes = np.argsort(self._data[index])
        nd = np_data[:, indexes].tolist()
        if inplace:
            self._data = nd
        return nd

    def raw_data(self, cols=None):
        return self._get_subdata(cols)

    def as_dataframe(self, cols=None, raw=True):
        if len(self._data) < 1:
            return pd.DataFrame()

        if cols is None:
            columns = self._labels if len(self._labels) > 0 else None
        else:
            columns = [self._labels[i] for i in self.label_to_col_index(cols)]
        return pd.DataFrame(self.as_array(cols=cols).T, columns=columns)

    def as_array(self, cols=None):
        return np.array(self._get_subdata(cols=cols))

    def get_indexes(self, **kwargs):
        keys = list(kwargs.keys())
        ci = self.label_to_col_index(keys)
        data = self.as_array(cols=ci)
        for i, k in enumerate(keys):
            data[i] = data[i] == kwargs[k]
        return np.nonzero(np.prod(data, axis=0))[0].tolist()

    def _get_subdata(self, cols=None):
        if cols is None:
            return self._data
        else:
            ci = self.label_to_col_index(cols)
            return [self._data[i] for i in ci]

    def insert_at_index(self, index=None, **kwargs):
        if index is None:
            index = self._current_index

        keys = list(kwargs.keys())
        for key in keys:
            if key not in self._labels:
                self._labels.append(key)
                self._data.append([None] * (self._current_index + 1))

        ci = self.label_to_col_index(keys)
        dataa = np.array([kwargs[k] for k in keys])
        npa = self.as_array()
        if isinstance(index, int):
            index = [index]
        npa[ci, index] = dataa
        self._data = npa.tolist()

    def get_nearest(self, **kwargs):
        keys = list(kwargs.keys())
        ci = self.label_to_col_index(keys)
        dataa = np.array([[kwargs[k] for k in keys]]).T
        a = self.as_array(cols=keys)
        dist = np.linalg.norm(a - dataa, axis=0)
        return np.where(dist == dist.min())[0].tolist()

    def save(self, path=None,only_if_data=True):
        if len(self._data)>0:
            self._save_rules_last_save_line = len(self._data[0])
        else:
            self._save_rules_last_save_line = 0
            if only_if_data:
                return None

        if path is None:
            path = self._filename.format(time=pd.Timestamp(timer(), unit=time_unit).strftime('%Y_%m_%d_%H_%M_%S'),
                                         format="csv")
        path = os.path.join(self._folder, path)

        save_df = self.as_dataframe()

        if self._pre_save_path:
            os.remove(self._pre_save_path)
        save_df.to_csv(path, index=None)
        self._pre_save_path = path

        return path

    def backup_rules(self, min_time=None, max_time=None, min_data=None, max_data=None, backup_count=None):
        if max_time is not None:
            max_time = int(max_time)
        if min_time is not None:
            min_time = int(min_time)
        if max_time is not None:
            self._backup_max_time = max_time
        if min_time is not None:
            self._backup_min_time = min_time
        if self._backup_max_time < self._backup_min_time:
            self._backup_min_time = self._backup_max_time

        if backup_count:
            backup_count = int(backup_count)
            assert backup_count >= 0, "backupcount cannot be negativ"
            self._backupcount = backup_count

        if max_data is not None:
            max_data = int(max_data)
        if min_data is not None:
            min_data = int(min_data)
        if max_data is not None:
            self._backup_max_data = max_data
        if min_data is not None:
            self._backup_min_data = min_data
        if self._backup_max_data < self._backup_min_data:
            self._backup_min_data = self._backup_max_data

        if self._backup_max_time < self._backup_min_time:
            self._backup_min_time = self._backup_max_time

        self._check_backup()

    def create_backup(self):
        path=os.path.join(self._folder, self._filename.format(
            time=pd.Timestamp(timer(), unit=time_unit).strftime('%Y_%m_%d_%H_%M_%S_%f'), format="bu"))

        save_df = self.as_dataframe()

        save_df.to_csv(path, index=None)

        files = list(
            filter(os.path.isfile, glob.glob(os.path.join(self._folder, self._filename.format(time="*", format="bu")))))

        if len(files) >= self._backupcount:
            files.sort(key=lambda x: os.path.getmtime(x))
            for f in files[:-self._backupcount]:
                os.remove(f)

        self._last_backup_time = time.time()
        self._last_backup_index = self._current_index

    def _check_backup(self):
        t = time.time()
        if self._backupcount > 0:
            if (
                    self._last_backup_time + self._backup_max_time <= t and self._last_backup_index + self._backup_min_data <= self._current_index) or \
                    (
                            self._last_backup_index + self._backup_max_data <= self._current_index and self._last_backup_time + self._backup_min_time <= t):
                self.create_backup()

    def saving_rules(self, max_lines=None):
        if max_lines is not None:
            max_lines = int(max_lines)
            assert max_lines > 0
            self._save_rules_max_lines = max_lines

        self._check_save()

    def _check_save(self):
        if len(self._data) > 0:
            if len(self._data[0]) - self._save_rules_last_save_line >= self._save_rules_max_lines:
                self.save()

    def reset(self):
        self._start_time = timer()
        time.sleep(0.1)
        timestamp = pd.Timestamp(self._start_time, unit=time_unit).strftime('%Y_%m_%d_%H_%M_%S_%f')
        self._filename = f"{self._basename}_{timestamp}__{{time}}.{{format}}"
        self._data = []
        self._labels = []
        self._current_index = -1
        self._save_rules_last_save_line = 0
        self._last_backup_time = time.time()
        self._last_backup_index = self._current_index
        self._pre_save_path=None

    def _sort_columns(self):
        l, d = zip(*sorted(zip(self._labels, self._data)))
        self._labels = list(l)
        self._data = list(d)


class TimeSeriesDataRecorder(DataRecorder):
    def __init__(self, resolution=10 ** -3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resolution = 10 ** -3
        self.set_resolution(resolution)

    def reset(self):
        super().reset()
        self._last_record_time = None
        self._last_t = -np.inf

    def start_timer(self):
        assert self._start_time == None, "timer already running"
        self._start_time = timer()

    def get_start_time(self):
        return self._start_time

    def set_resolution(self, seconds=10 ** -3):
        self._resolution = seconds*res_fac

    def data_point(self, **kwargs):
        if not self._start_time:
            self.start_timer()
        t = timer()
        if (t - self._last_t) >= self._resolution:
            super().data_point(time=t, **kwargs)
            self._last_t = t
        else:
            self.insert_at_index(**kwargs)

    def as_dataframe(self, cols=None, as_delta=False, as_date=True, raw=False):
        df = super().as_dataframe(cols=cols)
        if raw:
            return df

        if as_date and 'time' in df.columns:
            if as_delta:
                df['time'] = df['time'].apply(lambda x: pd.Timedelta(x - self._start_time, unit=time_unit))
            else:
                df['time'] = df['time'].apply(lambda x: pd.Timestamp(x, unit=time_unit))
        return df

    def as_array(self, cols=None, as_delta=False):
        try:
            n = super().as_array(cols=cols)
            if as_delta:
                n[0] = n[0] - self._start_time
        except Exception as e:
            raise e
        return n


    def _sort_columns(self):
        l, d = zip(*sorted(zip(self._labels[1:], self._data[1:])))
        self._labels = [self._labels[0]] + list(l)
        self._data = [self._data[0]] + list(d)
