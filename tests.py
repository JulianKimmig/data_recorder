import datetime
import os
import tempfile
import time
import unittest
from random import random

import numpy
import pandas
from PIL import Image

from datarecorder import TimeSeriesDataRecorder, DataRecorder


class SimpleDataRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dr = DataRecorder()
        for i in range(0,10,2):
            self.dr.data_point(x=i,y=i*i,z=numpy.sin(i),r=(random()+random())*i*i,c=10)
        for i in range(1,10,2):
            self.dr.data_point(x=i,y=i*i,z=numpy.sin(i),r=(random()+random())*i*i,c=10)


    def test_get_pandas(self):
        print(self.dr.as_dataframe(cols=[1,"c","z",1]))

    def test_get_numpy(self):
        self.dr.sort(index=0)
        assert numpy.all(self.dr.as_array(cols=["x","y"])==numpy.array([[x,x*x] for x in range(10)]).T)

    def test_sort(self):
        nd = self.dr.sort(index=0,inplace=False)
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

class TimeSeriesDataRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dr = TimeSeriesDataRecorder()

    def test_time(self):
        self.dr.data_point(x=1)
        self.dr.set_resolution(0.2)
        self.dr.data_point(y=2)
        self.dr.set_resolution(0)
        assert (self.dr.as_array().shape[1] == 1)
        self.dr.data_point(y=3)
        self.dr.data_point(x=0)
        assert (self.dr.as_array().shape[1] == 3)

        print(self.dr.as_dataframe())

class SpectraRecorder(unittest.TestCase):
    def setUp(self) -> None:
        self.source_image = tempfile.mkstemp()[1]
        self.target_image = tempfile.mkstemp()[1]
        self.data_recorder = TimeSeriesDataRecorder()
        self.spectra = None

    def test_generate_spectra(self):
        self.assertEqual(True, False)

    def test_compare_spectra(self):
        pass

    def test_spectra_compare_images(self):
        pic1 = numpy.asarray(Image.open(self.source_image))
        pic2 = numpy.asarray(Image.open(self.target_image))
        print(pic1,pic2)

    def tearDown(self) -> None:
        os.remove(self.source_image)
        os.remove(self.target_image)




if __name__ == '__main__':
    unittest.main()
