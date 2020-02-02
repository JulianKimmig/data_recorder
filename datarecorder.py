import glob
import os
import time

try:
    timer=time.time_ns
    time_unit = "ns"
except:
    timer=time.time
    time_unit = "s"

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
        self._save_rules_max_lines = 10**4
        self._save_rules_keep_lines = int(self._save_rules_max_lines / 2)


    def label_to_col_index(self, col_array):
        col_indexes=col_array.copy()
        for i,j in enumerate(col_indexes):
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


    def plot2d(self, target_file=None, x=None, y=None,with_labels=True):

        if y is None:
            y = [i for i in range(len(self._labels))]

        assert len(y) > 0 , "Nothing to plot"

        if x is not None:
            if isinstance(x, int):
                assert x < len(self._labels), f"x: {x} not in the range of labels"
                xi = x
            else:
                assert x in self._labels, f"x: {x} is not in labels"
                xi = self._labels.index(x)

            x_label = self._labels[xi]
            x = self._data[xi]

        else:
            x = np.arange(len(self._data[0]))
            x_label="data point #"
        
        y = self.label_to_col_index(y)

        for i in y:
            plt.plot(x, self._data[i], label=self._labels[i])
        if with_labels:
            plt.xlabel(x_label)
            plt.legend()

        if target_file:
            plt.savefig(target_file)
        return plt

    def sort(self, index, inplace=True):
        np_data=self.as_array()
        index = self.label_to_col_index([index])[0]
        indexes = np.argsort(self._data[index])
        nd=np_data[:,indexes].tolist()
        if inplace:
            self._data = nd
        return nd

    def raw_data(self,cols=None):
        return self._get_subdata(cols)

    def as_dataframe(self, cols=None, raw=True):
        if len(self._data) < 1:
            return pd.DataFrame()

        if cols is None:
            columns = self._labels if len(self._labels) > 0 else None
        else:
            columns = [self._labels[i] for i in self.label_to_col_index(cols)]
        return pd.DataFrame(self.as_array(cols=cols).T,columns=columns)

    def as_array(self,cols=None):
        return np.array(self._get_subdata(cols=cols))

    def get_indexes(self, **kwargs):
        keys = list(kwargs.keys())
        ci = self.label_to_col_index(keys)
        data=self.as_array(cols=ci)
        for i,k in enumerate(keys):
            data[i] = data[i] == kwargs[k]
        return np.nonzero(np.prod(data,axis=0))[0].tolist()

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
        dataa=np.array([kwargs[k] for k in keys])
        npa = self.as_array()
        if isinstance(index,int):
            index = [index]
        npa[ci,index] = dataa
        self._data = npa.tolist()

    def get_nearest(self, **kwargs):
        keys = list(kwargs.keys())
        ci = self.label_to_col_index(keys)
        dataa=np.array([[kwargs[k] for k in keys]]).T
        a = self.as_array(cols=keys)
        dist = np.linalg.norm(a-dataa,axis=0)
        return np.where(dist == dist.min())[0].tolist()

    def save(self, path=None, full=False, without_drop=False):
        if path is None:
            path = self._filename.format(time=pd.Timestamp(timer(), unit=time_unit).strftime('%Y_%m_%d_%H_%M_%S'),
                                         format="csv")
        path = os.path.join(self._folder, path)
        if os.path.exists(path):
            i = 2
            inipath = path
            while os.path.exists(path):
                path = "{0}__{2}.{1}".format(*inipath.rsplit('.', 1) + [i])
                i += 1

        new_df = self.as_dataframe(raw=True)
        keep_lines = min(self._save_rules_keep_lines, len(new_df))
        fromsave = 0 if full else self._save_rules_last_save_line

        save_df = self.as_dataframe()[fromsave:]
        save_df.to_csv(path, index=None)

        if not without_drop:
            self._data = new_df[-keep_lines:].values.T.tolist()
            self._save_rules_last_save_line = keep_lines

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
        self.save(path=os.path.join(self._folder, self._filename.format(
            time=pd.Timestamp(timer(), unit=time_unit).strftime('%Y_%m_%d_%H_%M_%S'), format="bu")), full=True,
                  without_drop=True)
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

    def saving_rules(self, max_lines=None, keep_lines=None):
        if max_lines is not None:
            max_lines = int(max_lines)
            assert max_lines > 0
            self._save_rules_max_lines = max_lines

        if keep_lines is not None:
            keep_lines = int(keep_lines)
            assert keep_lines >= 0
            self._save_rules_keep_lines = keep_lines

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

    def _sort_columns(self):
        l, d = zip(*sorted(zip(self._labels, self._data)))
        self._labels = list(l)
        self._data = list(d)


class TimeSeriesDataRecorder(DataRecorder):
    def __init__(self, resolution=10 ** -3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resolution = 10**-3
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

    def set_resolution(self,seconds=10**-3):
        self._resolution=seconds

    def data_point(self, **kwargs):
        if not self._start_time:
            self.start_timer()
        t=timer()
        if (t-self._last_t) >= self._resolution:
            super().data_point(time=t,**kwargs)
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

    def as_array(self,cols=None,as_delta=False):
        try:
            n = super().as_array(cols=cols)
            if as_delta:
                n[0] = n[0] - self._start_time
        except Exception as e:
            print(n)
            raise e
        return n

    def plot2d(self, x=None, y=None, **kwargs):
        if x is None:
            x = "time"

        if y is None:
            y = [i for i, l in enumerate(self._labels) if l != "time"]
        return super().plot2d(x=x, y=y, **kwargs)

    def _sort_columns(self):
        l, d = zip(*sorted(zip(self._labels[1:], self._data[1:])))
        self._labels = [self._labels[0]] + list(l)
        self._data = [self._data[0]] + list(d)
