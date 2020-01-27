import datetime
import time

try:
    timer=time.time_ns
except:
    timer=time.time


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataRecorder():
    def __init__(self):
        self._data = []
        self._lables = []
        self._current_index = -1

    def label_to_col_index(self, col_array):
        col_indexes=col_array.copy()
        for i,j in enumerate(col_indexes):
            if isinstance(j, int):
                assert j < len(self._lables), f"y: {j} not in the range of labels"
            else:
                assert j in self._lables, f"y: {j} is not in labels"
                col_indexes[i] = self._lables.index(j)
        return col_indexes

    def data_point(self, **kwargs):
        self._current_index += 1
        for key in kwargs.keys():
            if key not in self._lables:
                self._lables.append(key)
                self._data.append([None] * self._current_index)

        nd = [kwargs.get(l, None) for l in self._lables]
        for i, d in enumerate(nd):
            self._data[i].append(d)

    def plot2d(self, target_file=None, x=None, y=None,with_labels=True):

        if y is None:
            y = [i for i in range(len(self._lables))]

        assert len(y) > 0 , "Nothing to plot"

        if x is not None:
            if isinstance(x, int):
                assert x < len(self._lables), f"x: {x} not in the range of labels"
                xi = x
            else:
                assert x in self._lables, f"x: {x} is not in labels"
                xi = self._lables.index(x)

            x_label = self._lables[xi]
            x = self._data[xi]

        else:
            x = np.arange(len(self._data[0]))
            x_label="data point #"
        
        y = self.label_to_col_index(y)

        for i in y:
            plt.plot(x, self._data[y[i]],label=self._lables[y[i]])
        if with_labels:
            plt.xlabel(x_label)
            plt.legend()

        if target_file:
            plt.savefig(target_file)
        return plt

    def sort(self, index, inplace=True):
        np_data=self.as_array()
        indexes = np.argsort(self._data[index])
        nd=np_data[:,indexes].tolist()
        if inplace:
            self._data = nd
        return nd

    def raw_data(self,cols=None):
        return self._get_subdata(cols)

    def as_dataframe(self,cols=None):
        if cols is None:
            columns=self._lables if len(self._lables) > 0 else None
        else:
            columns = [self._lables[i] for i in self.label_to_col_index(cols)]
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
            if key not in self._lables:
                self._lables.append(key)
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


class TimeSeriesDataRecorder(DataRecorder):
    def __init__(self,resolution=10**-3):
        super().__init__()
        self._start_time = None
        self._last_record_time = None
        self._resolution = 10**-3
        self.set_resolution(resolution)
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

    def as_dataframe(self,cols=None,as_delta=False,as_date=True):
        if cols is None:
            columns=self._lables if len(self._lables) > 0 else None
        else:
            columns = [self._lables[i] for i in self.label_to_col_index(cols)]
        n = self.as_array(cols=cols,as_delta=as_delta)

        if as_date:
            if as_delta:
                n[0] = np.array([pd.Timedelta(xi, unit='s') for xi in n[0]])
            else:
                n[0] = np.array([pd.Timestamp(xi, unit='s') for xi in n[0]])
        return pd.DataFrame(n.T,columns=columns)

    def as_array(self,cols=None,as_delta=False):
        n = super().as_array(cols=cols)
        if as_delta:
            n[0]=n[0]-self._start_time
        return n
