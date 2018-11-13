# ilang - Inference Language
# Stefano Pedemonte
# Oct. 2013, Alessandria, Italy

from __future__ import absolute_import, print_function
__all__ = ['DisplayPylab']

import numpy as np

try:
    import pylab
except:
    print("Please install Pylab")
    has_pylab = False
else:
    has_pylab = True
    # pylab.ion()
    from colorsys import hsv_to_rgb
try:
    import matplotlib.pyplot as plt
    plt.ion()
except:
    pass


class DisplayPylab():
    """Display information about the samples stored in a tracer. """

    def __init__(self, tracer):
        self._tracer = tracer

    def plot_mean(self, figure=None, cmap=None, title="Mean", log_scale=False, interpolation='none'):
        mean = self._tracer.mean()
        if mean is None:
            return False
        else:
            if not figure:
                pass  # figure = pylab.figure()
            else:
                pylab.figure(figure.number)
            if cmap is None:
                cmap = pylab.cm.Greys_r
            if log_scale:
                mean = np.log(mean - mean.min() + 1e-10)
            pylab.imshow(mean, cmap, interpolation=interpolation)
            pylab.colorbar()
            pylab.title(title)
            pylab.draw()
        return figure

    def plot_variance(self, figure=None, cmap=None, title="Variance", log_scale=False, interpolation='none'):
        variance = self._tracer.variance()
        if variance is None:
            return False
        else:
            if not figure:
                pass  # figure = pylab.figure()
            else:
                pylab.figure(figure.number)
            if cmap is None:
                cmap = pylab.cm.Greys_r
            if log_scale:
                variance = np.log(variance - variance.min() + 1e-10)
            pylab.imshow(variance, cmap, interpolation=interpolation)
            pylab.title(title)
            pylab.draw()
        return figure

    def plot_sample_index(self, index, figure=None, cmap=None, title=None, log_scale=False, interpolation='none'):
        samples = self._tracer.get_samples()
        if samples is None:
            return False
        sample = samples[index]
        if title is None:
            title = "Sample %d" % index
        return self.plot_sample_value(sample, figure, cmap, title, log_scale, interpolation)

    def plot_sample_value(self, sample, figure=None, cmap=None, title="Sample", log_scale=False, interpolation='none'):
        if not figure:
            pass  # figure = pylab.figure()
        else:
            pylab.figure(figure.number)
        if cmap is None:
            cmap = pylab.cm.Greys_r
        pylab.imshow(sample, cmap, interpolation=interpolation)
        pylab.title(title)
        pylab.draw()
        return figure

    def plot_mean_roi(self, roi, figure=None, cmap=None, title="Mean ROI", log_scale=False, interpolation='none'):
        mean = self._tracer.mean_roi(roi)
        if mean is None:
            return False
        else:
            if not figure:
                pass  # figure = pylab.figure()
            else:
                pylab.figure(figure.number)
            if cmap is None:
                cmap = pylab.cm.Greys_r
            if log_scale:
                mean = np.log(mean - mean.min() + 1e-10)
            pylab.imshow(mean, cmap, interpolation=interpolation)
            pylab.title(title)
            pylab.draw()
        return figure

    def plot_variance_roi(self, roi, figure=None, cmap=None, title="Variance ROI", log_scale=False,
                          interpolation='none'):
        variance = self._tracer.variance_roi(roi)
        if variance is None:
            return False
        else:
            if not figure:
                pass  # figure = pylab.figure()
            else:
                pylab.figure(figure.number)
            if cmap is None:
                cmap = pylab.cm.Greys_r
            if log_scale:
                variance = np.log(variance - variance.min() + 1e-10)
            pylab.imshow(variance, cmap, interpolation=interpolation)
            pylab.title(title)
            pylab.draw()
        return figure

    def plot_covariance(self, figure=None, cmap=None, title="Covariance", log_scale=False, interpolation='none'):
        covariance = self._tracer.covariance()
        if covariance is None:
            return False
        else:
            if not figure:
                pass  # figure = pylab.figure()
            else:
                pylab.figure(figure.number)
            if cmap is None:
                cmap = pylab.cm.Greys_r
            if log_scale:
                covariance = np.log(covariance - covariance.min() + 1e-10)
            pylab.imshow(covariance, cmap, interpolation=interpolation)
            pylab.title(title)
            pylab.draw()
        return figure

    def plot_covariance_line(self, xy, covariance=None, figure=None, cmap=None, title="Covariance", log_scale=False,
                             interpolation='none'):
        covariance_line = self._tracer.covariance_line(xy, covariance)
        if covariance_line is None:
            return False
        else:
            if not figure:
                pass  # figure = pylab.figure()
            else:
                pylab.figure(figure.number)
            if cmap is None:
                cmap = pylab.cm.Greys_r
            if log_scale:
                covariance_line = np.log(covariance_line - covariance_line.min() + 1e-10)
            pylab.imshow(covariance_line, cmap, interpolation=interpolation)
            pylab.title(title)
            pylab.draw()
        return figure

    def plot_histogram_bivariate(self, coordinates_A, coordinates_B, n_bins=(100, 100), range_plot=None, figure=None,
                                 cmap=None, title="Histogram", log_scale=False, interpolation='bicubic', subplot=None):
        histogram, x, y = self._tracer.histogram_bivariate(coordinates_A, coordinates_B, n_bins, range_plot)
        if histogram is None:
            return False
        else:
            if not figure:
                pass  # figure = pylab.figure()
            else:
                pylab.figure(figure.number)
            if subplot is not None:
                pylab.subplot(*subplot)
            if cmap is None:
                cmap = pylab.cm.Greys_r
            if log_scale:
                histogram = np.log(histogram + 1e-10)
            pylab.imshow(histogram, cmap=cmap, interpolation=interpolation)
            pylab.title(title)
            pylab.draw()
        return figure

    def plot_histogram(self, coordinates, n_bins=100, range_plot=None, figure=None, alpha_line=0.8, alpha_fill=0.2,
                       color_line=None, color_fill=None, title="Histogram", true_value=None, estimated_value=None,
                       subplot=None):
        histogram, x = self._tracer.histogram(coordinates, n_bins, range_plot)
        if histogram is None:
            return False
        else:
            if color_line is None:
                color_line = hsv_to_rgb(0.0, 0.85, 1)
            if color_fill is None:
                color_fill = hsv_to_rgb(0.0, 0.85, 1)
            if not figure:
                pass  # figure = pylab.figure()
            else:
                pylab.figure(figure.number)
            if subplot is not None:
                pylab.subplot(*subplot)
            pylab.plot(x[0:-1], histogram, alpha=alpha_line, color=color_line)
            pylab.fill_between(x[0:-1], 0, histogram, alpha=alpha_fill, facecolor=color_fill)

            mean_value = self._tracer.mean_at_location(coordinates)
            pylab.axvline(x=mean_value, ymin=0.0, ymax=0.99, alpha=alpha_line, color=color_line, linewidth=1,
                          linestyle='dashed')

            if true_value is not None:
                if not isinstance(true_value, type(1)):
                    true_value = true_value[coordinates[0], coordinates[1]]
                pylab.axvline(x=true_value, ymin=0.0, ymax=0.99, alpha=alpha_line, color=color_line, linewidth=1,
                              linestyle='solid')
            if estimated_value is not None:
                if not isinstance(true_value, type(1)):
                    estimated_value = estimated_value[coordinates[0], coordinates[1]]
                pylab.axvline(x=estimated_value, ymin=0.0, ymax=0.99, alpha=alpha_line, color=color_line, linewidth=1,
                              linestyle='dotted')
            pylab.title(title)
            pylab.draw()
        return figure

    def plot_histogram_roi_integral(self, roi, n_bins=100, range_plot=None, figure=None, alpha_line=0.8, alpha_fill=0.2,
                                    color_line=None, color_fill=None, title="ROI integral", true_value=None,
                                    estimated_value=None):
        histogram, x = self._tracer.histogram_roi_integral(roi, n_bins, range_plot)
        if histogram is None:
            return False
        else:
            if color_line is None:
                color_line = hsv_to_rgb(0.0, 0.85, 1)
            if color_fill is None:
                color_fill = hsv_to_rgb(0.0, 0.85, 1)
            if not figure:
                pass  # figure = pylab.figure()
            else:
                pylab.figure(figure.number)
            pylab.plot(x[0:-1], histogram, alpha=alpha_line, color=color_line)
            pylab.fill_between(x[0:-1], 0, histogram, alpha=alpha_fill, facecolor=color_fill)
            figure.hold(1)
            if true_value is not None:
                if not isinstance(true_value, type(1)):
                    true_value = self._tracer.extract_roi_from_sample(roi, true_value).sum()
                pylab.axvline(x=true_value, ymin=0.0, ymax=0.99, color=color_line, alpha=alpha_line, linewidth=1)
            if estimated_value is not None:
                if not isinstance(true_value, type(1)):
                    estimated_value = self._tracer.extract_roi_from_sample(roi, estimated_value).sum()
                pylab.axvline(x=estimated_value, ymin=0.0, ymax=0.99, color=color_line, alpha=alpha_line, linewidth=1,
                              linestyle='dashed')
            pylab.title(title)
            pylab.draw()
        return figure

    def plot_profile(self, axis, index, n_bins=100, range_plot=None, true_value=None, estimated_value=None, figure=None,
                     cmap=None, color_line=None, title="Profile"):
        # FIXME: implementation is 2-D only, extend to 3-D
        if axis != 0 and axis != 1:
            raise ("axis: 0 or 1")
        if cmap is None:
            cdict = {'red': ((0.0, 1.0, 1.0), (1.0, 0.7, 0.7)), 'green': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
                     'blue': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0))}
            cmap = pylab.matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
        shape = self._tracer.shape()
        n_points = shape[1 - axis]
        M = np.zeros((n_points, n_bins))
        if range_plot is None:
            range_plot = (self._tracer.min(), self._tracer.max())
        for i in range(n_points):
            if axis == 0:
                M[i, :] = self._tracer.histogram((index, i), n_bins=100, range_histogram=range_plot)[0]
            else:
                M[i, :] = self._tracer.histogram((i, index), n_bins=100, range_histogram=range_plot)[0]
        if axis == 0:
            p = self._tracer.get_samples()[:, index, :]
            if true_value is not None:
                p_t = true_value[index, :]
            if estimated_value is not None:
                p_e = estimated_value[index, :]
        else:
            p = self._tracer.get_samples()[:, :, index]
            if true_value is not None:
                p_t = true_value[:, index]
            if estimated_value is not None:
                p_e = estimated_value[:, index]
        p_m = np.mean(p, 0)
        if color_line is None:
            color_line = hsv_to_rgb(0.2, 0.0, 0.3)
        if not figure:
            pass  # figure = pylab.figure()
        else:
            pylab.figure(figure.number)
        pylab.hold(1)
        pylab.plot((n_bins / (range_plot[1] - range_plot[0])) * p_m, color=color_line, linewidth=1, linestyle='dashed')
        if true_value is not None:
            pylab.plot((n_bins / (range_plot[1] - range_plot[0])) * p_t, color=color_line, linewidth=1,
                       linestyle='solid')
        if estimated_value is not None:
            pylab.plot((n_bins / (range_plot[1] - range_plot[0])) * p_e, color=color_line, linewidth=1,
                       linestyle='dotted')
        pylab.imshow(M.transpose(), cmap=cmap, origin="lower", aspect="auto")
        pylab.title(title)
        pylab.draw()

    def plot_autocorrelation(self, coordinates, max_lag=None, figure=None, alpha_line=0.8, color_line=None,
                             title="Autocorrelation"):
        autocorrelation = self._tracer.autocorrelation(coordinates)
        if autocorrelation is None:
            return False
        else:
            if color_line is None:
                color_line = hsv_to_rgb(0.0, 0.85, 1)
            if not figure:
                pass  # figure = pylab.figure()
            else:
                pylab.figure(figure.number)
            if max_lag is None:
                max_lag = self._tracer.size()
            elif max_lag >= self._tracer.size() or max_lag <= 0:
                max_lag = self._tracer.size()
            pylab.plot(autocorrelation[0:max_lag], alpha=alpha_line, color=color_line)

            pylab.title(title)
            pylab.draw()
        return figure

    def plot_acceptance_ratio(self, figure=None, alpha_line=0.8, alpha_fill=0.2, color_line=None, color_fill=None,
                              title="Acceptance Ratio"):
        return self._plot_simple_curves(self._tracer.get_acceptance_ratio(), figure, alpha_line, alpha_fill, color_line,
                                        color_fill, title)

    def plot_accepted(self, figure=None, alpha_line=0.8, alpha_fill=0.2, color_line=None, color_fill=None,
                      title="Accepted"):
        return self._plot_simple_curves(self._tracer.get_accepted(), figure, alpha_line, alpha_fill, color_line,
                                        color_fill, title)

    def plot_threshold(self, figure=None, alpha_line=0.8, alpha_fill=0.2, color_line=None, color_fill=None,
                       title="Threshold"):
        return self._plot_simple_curves(self._tracer.get_threshold(), figure, alpha_line, alpha_fill, color_line,
                                        color_fill, title)

    def _plot_simple_curves(self, curve, figure=None, alpha_line=0.8, alpha_fill=0.2, color_line=None, color_fill=None,
                            title=" "):
        if color_line is None:
            color_line = hsv_to_rgb(0.0, 0.85, 1)
        if color_fill is None:
            color_fill = hsv_to_rgb(0.0, 0.85, 1)
        if not figure:
            pass  # figure = pylab.figure()
        else:
            pylab.figure(figure.number)
        pylab.plot(curve, alpha=alpha_line, color=color_line)
        pylab.title(title)
        # pylab.draw()
        return figure
