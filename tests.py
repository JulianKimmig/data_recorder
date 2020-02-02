import glob
import os
import tempfile
import time
import unittest
from random import random

import numpy
import pandas

from datarecorder import TimeSeriesDataRecorder, DataRecorder


class SimpleDataRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dr = DataRecorder(folder=tempfile.mkdtemp(), basename="testrecorder")
        for i in range(0,10,2):
            self.dr.data_point(x=i,y=i*i,z=numpy.sin(i),r=(random()+random())*i*i,c=10)
        for i in range(1,10,2):
            self.dr.data_point(x=i,y=i*i,z=numpy.sin(i),r=(random()+random())*i*i,c=10)


    def test_get_pandas(self):
        print(self.dr.as_dataframe(cols=[1,"c","z",1]))

    def test_get_numpy(self):
        self.dr.sort(index="x")
        assert numpy.all(self.dr.as_array(cols=["x","y"])==numpy.array([[x,x*x] for x in range(10)]).T)

    def test_sort(self):
        nd = self.dr.sort(index=0,inplace=False)
        assert len(nd) == len(self.dr.raw_data()) and len(nd[0]) == len(self.dr.raw_data()[0])
        assert numpy.array_equal(numpy.array(nd)[0],self.dr.as_array()[0][numpy.argsort(self.dr.as_array()[0])])
        nd = self.dr.sort(index="x",inplace=False)
        assert len(nd) == len(self.dr.raw_data()) and len(nd[0]) == len(self.dr.raw_data()[0])
        assert numpy.array_equal(numpy.array(nd)[0],self.dr.as_array()[0][numpy.argsort(self.dr.as_array()[0])])
    def test_find_near(self):
        assert self.dr.get_nearest(c=10) == list(range(10))
        assert self.dr.get_nearest(x=3.4,y=15) == [2]

    def test_get_image(self):
        self.dr.sort(index=0,inplace=True)
        self.dr.plot2d()#.show()

    def test_insert_at_specific_point(self):
        assert self.dr.get_indexes(c=10,x=2) == [1]
        assert self.dr.get_indexes(y=3,x=2) == []
        assert self.dr.get_indexes(c=10) == list(range(10))
        self.dr.insert_at_index(self.dr.get_indexes(x=2),c=5,y=20)
        assert all(self.dr.as_array()[self.dr.label_to_col_index(["c","y"]),self.dr.get_indexes(x=2)] == [5,20])

    def test_save(self):
        savefile = self.dr.save()
        assert os.path.exists(savefile)
        os.remove(savefile)

    def test_backup(self):
        globsearch = os.path.join(self.dr._folder, self.dr._filename.format(time="*", format="bu"))
        print(globsearch)
        self.dr.backup_rules(min_time=0, max_time=0, min_data=0, max_data=100, backup_count=100)
        files = list(filter(os.path.isfile, glob.glob(globsearch)))
        for f in files:
            os.remove(f)
        for i in range(10):
            self.dr.data_point(b=1)
        files = list(filter(os.path.isfile, glob.glob(globsearch)))
        assert len(files) == 10, f"{len(files)} files found"
        for f in files:
            os.remove(f)

        self.dr.backup_rules(max_time=120)
        for i in range(10):
            self.dr.data_point(b=1)
        files = list(filter(os.path.isfile, glob.glob(globsearch)))
        assert len(files) == 0, f"{len(files)} files found"
        for f in files:
            os.remove(f)

        self.dr.backup_rules(max_data=1)
        for i in range(10):
            self.dr.data_point(b=1)
        files = list(filter(os.path.isfile, glob.glob(globsearch)))
        assert len(files) == 11, f"{len(files)} files found"
        for f in files:
            os.remove(f)

        self.dr.backup_rules(backup_count=3)
        for i in range(10):
            self.dr.data_point(b=1)
        files = list(filter(os.path.isfile, glob.glob(globsearch)))
        assert len(files) == 3, f"{len(files)} files found"
        for f in files:
            os.remove(f)

    def test_split_saving(self):
        self.dr.sort("x")
        globsearch = os.path.join(self.dr._folder, f"{self.dr._basename}_*.csv")
        self.dr.saving_rules(max_lines=5)
        for i in range(10):
            self.dr.data_point(x=i + 10)
        files = list(filter(os.path.isfile, glob.glob(globsearch)))
        assert len(files) == 3, f"{len(files)} files found"
        assert len(self.dr.as_dataframe()) == 20, f"dataframe len is {len(self.dr.as_dataframe())}"
        self.dr.saving_rules(keep_lines=1)
        for i in range(10):
            self.dr.data_point(x=i + 20)
        files = list(filter(os.path.isfile, glob.glob(globsearch)))
        assert len(files) == 5, f"{len(files)} files found"
        assert len(self.dr.as_dataframe()) == 1
        self.dr.save(full=True)
        files = list(filter(os.path.isfile, glob.glob(globsearch)))
        assert len(files) == 6, f"{len(files)} files found"
        df = pandas.DataFrame()
        for f in files:
            df = pandas.concat([df, pandas.read_csv(f)])
        df.reset_index(drop=True)
        arang = numpy.arange(31)
        arang[-1] = 29
        assert all(df["x"].values.astype(int) == arang), df["x"]

class TimeSeriesDataRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dr = TimeSeriesDataRecorder(folder=tempfile.mkdtemp())

    def test_time(self):
        self.dr.data_point(x=1)
        self.dr.set_resolution(0.2)
        self.dr.data_point(y=2)
        self.dr.set_resolution(0)
        time.sleep(10**-6)
        assert (self.dr.as_array().shape[1] == 1), f"{self.dr.as_array()}"
        self.dr.data_point(y=3)
        self.dr.data_point(x=0)
        assert (self.dr.as_array().shape[1] == 3)

    def tttest_write_speed(self):
        self.dr.backup_rules(min_time=0, min_data=0, max_data=100, backup_count=1)
        self.dr.saving_rules(max_lines=50, keep_lines=10 ** 6)
        self.dr.set_resolution(0)
        print(self.dr._folder)

        for i in range(5):
            self.dr.reset()
            start = time.time()
            for j in range(10 ** i):
                self.dr.data_point(x=j, i=i)
            self.dr.save(full=True)
            print(time.time() - start)

        for i in range(5):
            self.dr.reset()
            start = time.time()
            for j in range(10 ** i):
                self.dr.data_point(x=j, i=i, **{f"dp_{x}": x for x in range(10 ** 3)})
            self.dr.save(full=True)
            print(time.time() - start)

if __name__ == '__main__':
    unittest.main()
