# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 

from __future__ import absolute_import, print_function

__all__ = ['RamTracer']

from .Display import DisplayPylab
from .exceptions import *
from .verbose import *

import numpy


class Tracer():
    def __init__(self):
        self.reset_trace()

    def reset_trace(self):
        pass


class RamTracer(Tracer):
    """Stores the trace of a sampler in RAM memory. """

    def __init__(self, batch_size=1000):
        self._batch_size = batch_size  # allocate new memory evey '_batch_size' samples
        self.reset_trace()

    def reset_trace(self):
        self._samples = None  # samples
        self._shape = None  # shape of the sample array
        self._index = 0  # index of the last sample
        self._threshold = None
        self._accepted = None
        self._acceptance_ratio = None
        self._batch_index = 0
        self._size = 0  # size of the sample
        # Default display: 
        self.display = DisplayPylab(self)

    def get_trace(self):
        return self

    def empty(self):
        if self._samples is None or self._shape is None or self._index == 0:
            return True
        else:
            return False

    def shape(self):
        if self.empty():
            return (0,)
        else:
            return self._shape

    def size(self):
        if self.empty():
            return (0,)
        else:
            return self._size

    def change_batch_size(self, batch_size):
        self._batch_size = batch_size

    def set_shape(self, shape):
        if not numpy.asarray(shape).prod() == self._size:
            print("Not a valid shape, not compatible with sample size")
            return False
        else:
            self._shape = shape
            return True

    def scale(self, factor):
        for i in range(self._index):
            self._samples[i] = self._samples[i] * factor

    def new_sample(self, node_name, sample, parameters=None):
        if self._index == 0:
            # first sample, set the sample size and shape 
            self._size = sample.size
            self._shape = sample.shape
            self._samples = numpy.zeros((self._batch_size, self._size))
            self._threshold = numpy.zeros(self._batch_size)
            self._accepted = numpy.zeros(self._batch_size)
            self._acceptance_ratio = numpy.zeros(self._batch_size)
        else:
            if self._batch_index == 0:
                # allocate new batch
                self._samples = numpy.vstack((self._samples, numpy.zeros((self._batch_size, self._size))))
                #                self._threshold = numpy.hstack( (self._threshold,numpy.zeros(self._batch_size)) )
                #                self._accepted = numpy.hstack( (self._accepted,numpy.zeros(self._batch_size)) )
                #                self._acceptance_ratio = numpy.hstack( (self._acceptance_ratio,numpy.zeros(self._batch_size)) )
                #        print "Size: ",sample.size, sample.shape, self._size, self._shape
        self._samples[self._index, :] = sample.reshape((self._size,))
        #        self._threshold[self._index] = threshold
        #        self._accepted[self._index] = accepted
        #        self._acceptance_ratio[self._index] = acceptance_ratio
        self._index += 1
        self._batch_index += 1
        if self._batch_index == self._batch_size:
            self._batch_index = 0

    def get_samples(self):
        if self.empty():
            return None
        return self._samples[0:self._index, :].reshape((self._index,) + self._shape)

    def get_last_sample(self):
        return self._samples[self._index - 1, :].reshape(self._shape)

    def get_threshold(self):
        if self.empty():
            return None
        return self._threshold[0:self._index]

    def get_accepted(self):
        if self.empty():
            return None
        return self._accepted[0:self._index]

    def get_acceptance_ratio(self):
        if self.empty():
            return None
        return self._acceptance_ratio[0:self._index]

    def n_samples(self):
        if self.empty():
            return 0
        return self._index

    def min(self):
        if self.empty():
            return None
        return self._samples[0:self._index, :].min()

    def max(self):
        if self.empty():
            return None
        return self._samples[0:self._index, :].max()

    def mean(self):
        if self.empty():
            return None
        return numpy.mean(self._samples[0:self._index, :], 0).reshape(self._shape)

    def mean_at_location(self, coordinate):
        if self.empty():
            return None
        if isinstance((coordinate), type(1)):
            coordinate = (coordinate,)
        if len(coordinate) > 1:
            coordinate = numpy.ravel_multi_index(coordinate, self._shape)
        return numpy.mean(self.get_samples().reshape((self._index, self._size))[:, coordinate])

    def variance(self):
        if self.empty():
            return None
        return numpy.var(self._samples[0:self._index, :], 0).reshape(self._shape)

    def covariance(self):
        if self.empty():
            return None
        return numpy.cov(self._samples[0:self._index, :].T)

    def covariance_line(self, coordinates, covariance=None):
        if self.empty():
            return None
        if not covariance:
            covariance = self.covariance()
        if isinstance((coordinates), type(1)):
            coordinate = (coordinates,)
        if len(coordinates) > 1:
            coordinate = numpy.ravel_multi_index(coordinates, self._shape)
        return covariance[coordinate, :].reshape(self._shape)

    def histogram_bivariate(self, coordinate_A, coordinate_B, n_bins=(100, 100), range_histogram=None):
        if self.empty():
            return None
        if range_histogram is None:
            range_histogram = [(0, 1.2 * self.max()), (0, 1.2 * self.max())]
        if isinstance((coordinate_A), type(1)):
            coordinate_A = (coordinate_A,)
        if len(coordinate_A) > 1:
            coordinate_A = numpy.ravel_multi_index(coordinate_A, self._shape)
        if isinstance((coordinate_B), type(1)):
            coordinate_B = (coordinate_B,)
        if len(coordinate_B) > 1:
            coordinate_B = numpy.ravel_multi_index(coordinate_B, self._shape)
        return numpy.histogram2d(
            (self.get_samples().reshape((self._index, self._size))[:, coordinate_A]).reshape(self._index),
            (self.get_samples().reshape((self._index, self._size))[:, coordinate_B]).reshape(self._index), bins=n_bins,
            range=range_histogram)

    def histogram(self, coordinate, n_bins=100, range_histogram=None):
        if self.empty():
            return None
        if range_histogram is None:
            range_histogram = (0, 1.2 * self.max())
        if isinstance((coordinate), type(1)):
            coordinate = (coordinate,)
        if len(coordinate) > 1:
            coordinate = numpy.ravel_multi_index(coordinate, self._shape)
        return numpy.histogram(self.get_samples().reshape((self._index, self._size))[:, coordinate], bins=n_bins,
                               range=range_histogram)

    def histogram_roi_integral(self, roi, n_bins=100, range_histogram=None):
        if self.empty():
            return None
        roi_integral_samples = self.roi_integral_samples(roi)
        if range_histogram is None:
            range_histogram = (0, 1.1 * roi_integral_samples.max())
        return numpy.histogram(roi_integral_samples, bins=n_bins, range=range_histogram)

    def roi_samples(self, roi):
        x0, y0 = roi[0]
        x1, y1 = roi[1]
        return self.get_samples()[:, x0:x1, y0:y1]

    def roi_integral_samples(self, roi):
        return self.roi_samples(roi).sum(1).sum(1)

    def extract_roi_from_sample(self, roi, sample):
        x0, y0 = roi[0]
        x1, y1 = roi[1]
        return sample[x0:x1, y0:y1]

    def mean_roi(self, roi):
        if self.empty():
            return None
        roi_samples = self.roi_samples(roi)
        return numpy.mean(roi_samples[0:self._index, :], 0).reshape(roi_samples[0].shape)

    def variance_roi(self, roi):
        if self.empty():
            return None
        roi_samples = self.roi_samples(roi)
        return numpy.var(roi_samples[0:self._index, :], 0).reshape(roi_samples[0].shape)

    def autocorrelation(self, coordinate):
        if self.empty():
            return None
        if isinstance((coordinate), type(1)):
            coordinate = (coordinate,)
        if len(coordinate) > 1:
            coordinate = numpy.ravel_multi_index(coordinate, self._shape)
        samples = self.get_samples().reshape((self._index, self._size))[:, coordinate]
        samples = (samples - numpy.mean(samples)).reshape(samples.size)
        autocorr_f = numpy.correlate(samples, samples, mode='full')
        return autocorr_f[autocorr_f.size / 2:] / autocorr_f[autocorr_f.size / 2]
