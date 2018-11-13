# occiput
# Stefano Pedemonte
# Oct 2015
# Harvard University, Martinos Center for Biomedical Imaging
# Boston, MA, USA

from __future__ import absolute_import, print_function

__all__ = [
    'ProgressBar',
    'TriplanarView',
    'VolumeRenderer',
    'has_svgwrite',
    'has_ipy_table']

import uuid as _uuid
import h5py as _h5py
import numpy as np

from ...DisplayNode import DisplayNode
from IPython.display import HTML, display, Javascript
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt

from . import colors as C
from ..global_settings import is_gpu_enabled
from .ipynb import is_in_ipynb
from ipywidgets import interact, fixed, IntSlider, FloatProgress, FloatRangeSlider

import occiput_suite as _occiput_suite


class InstallationError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

# not compatible with jupyterLab, so we swtiched to ipywidget


class ProgressBar_legacy():
    def __init__(
            self,
            height='20',
            width='500',
            background_color=C.LIGHT_BLUE,
            foreground_color=C.BLUE,
            text_color=C.LIGHT_GRAY,
            title="processing ..."):
        self._inner_width = np.int(width) - 20
        self._percentage = 0.0
        self._divid = str(_uuid.uuid4())
        self.visible = False
        if is_in_ipynb():
            self.set_display_mode("ipynb")
        else:
            self.set_display_mode("text")
        self._pb = HTML(
            """
            <div>%s</div>
            <div style="border: 1px solid white; width:%spx; height:%s; background-color:%s; color:%s; font-weight:bold; ">
                <div id="%s" style="background-color:%s; width:0px; height:%spx;"> %s </div>
            </div>
            """ % (title, width, height, background_color, text_color, self._divid, foreground_color, height,
                   str(self._percentage)))

    def show(self):
        if self.mode == "ipynb":
            display(self._pb)
        self.visible = True

    def set_display_mode(self, mode="ipynb"):
        self.mode = mode

    def set_percentage(self, percentage):
        if not self.visible:
            self.show()
        if percentage < 0.0:
            percentage = 0.0
        if percentage > 100.0:
            percentage = 100.0
        percentage = int(percentage)
        self._percentage = percentage
        if self.mode == "ipynb":
            display(
                Javascript(
                    "$('div#%s').width('%i%%')" %
                    (self._divid, percentage)))
            display(
                Javascript(
                    "$('div#%s').text('%s')" %
                    (self._divid, str(percentage))))
        else:
            print("%2.1f / 100" % percentage)

    def get_percentage(self):
        return self._percentage


class ProgressBar():
    def __init__(self, color=C.BLUE, title="Processing:"):
        self._percentage = 0.0
        self.visible = False
        if is_in_ipynb():
            self.set_display_mode("ipynb")
        else:
            self.set_display_mode("text")
        if color == C.BLUE:
            color_style = ''
        elif color == C.LIGHT_BLUE:
            color_style = 'info'
        elif color == C.RED:
            color_style = 'danger'
        elif color == C.LIGHT_RED:
            color_style = 'warning'
        elif color == C.GREEN:
            color_style = 'success'
        else:
            print('Unavailable color code. Using default "BLUE", instead. ')
            color_style = ''

        self._pb = FloatProgress(
            value=0.0,
            min=0,
            max=100.0,
            step=0.1,
            description=title,
            bar_style=color_style,
            orientation='horizontal'
        )

    def show(self):
        if self.mode == "ipynb":
            display(self._pb)
        self.visible = True

    def set_display_mode(self, mode="ipynb"):
        self.mode = mode

    def set_percentage(self, percentage):
        if not self.visible:
            self.show()
        if percentage < 0.0:
            percentage = 0.0
        if percentage > 100.0:
            percentage = 100.0
        percentage = int(percentage)
        self._percentage = percentage
        if self.mode == "ipynb":
            self._pb.value = self._percentage
        else:
            print("%2.1f / 100" % percentage)

    def get_percentage(self):
        return self._percentage


# TODO: add class for image fusion built on top of TriplanarView (slider
# for alpha channel)

class TriplanarView:
    """
    Inspect a 3D or 4D data volume, with 3 planar view in Axial,
    Sagittal, and Coronal direction.
    You can pass a np.ndarray, or an occiput 'ImageND' object,
    or a 'PET_Projection' object.

    Usage
    -----
    V = TriplanarView(volume)
    V.show(clim=(0,15), colormap='color')
    """

    def __init__(self, volume, autotranspose=False):

        if isinstance(volume, np.ndarray):
            if autotranspose:
                if (volume.ndim == 3) or (volume.shape[3] == 1):
                    volume = volume.transpose(1, 0, 2)
                    volume = np.flip(volume, 2)  # U-D
                else:
                    volume = volume.transpose(2, 1, 3, 0)
                    volume = np.flip(volume, 2)  # U-D
            self.data = volume
            self.plottype = 'image'
        elif isinstance(volume, _occiput_suite.occiput.Core.Core.ImageND):
            volume = volume.data
            if autotranspose:
                if (volume.ndim == 3) or (volume.shape[3] == 1):
                    volume = volume.transpose(1, 0, 2)
                    volume = np.flip(volume, 2)  # U-D
                else:
                    volume = volume.transpose(2, 1, 3, 0)
                    volume = np.flip(volume, 2)  # U-D
            self.data = volume
            self.plottype = 'image'
        elif isinstance(volume, _occiput_suite.occiput.Reconstruction.PET.PET_Projection):
            volume = volume.to_nd_array()
            if autotranspose:
                volume = volume.transpose(0, 2, 3, 1)
            self.data = volume
            self.plottype = 'projection'
        else:
            print("Data format unknown")
            return

        # initialise where the image handles will go
        self.image_handles = None

    def show(self, colormap=None, figsize=(20, 9), clim=None, **kwargs):
        """
        Plot volumetric data.

        Args
        ----
            colormap : str
                    - 'color': show a selection of colorful colormaps
                    - 'mono': show a selection of monochromatic colormaps
                    - <cmap name>: pick one from matplotlib list
            clim : list | tuple
                    lower and upper bound for colorbar (you will have a slider
                    to better refine it later)
            figsize : list | tuple
                    The figure height and width for matplotlib, in inches.
            mask_background : bool
                    Whether the background should be masked (set to NA).
                    This parameter only works in conjunction with the default
                    plotting function (`plotting_func=None`). It finds clusters
                    of values that round to zero and somewhere touch the edges
                    of the image. These are set to NA. If you think you are
                    missing data in your image, set this False.
        """

        cmp_def = ['viridis']
        cmp_monochrome = ['binary', 'Greys', 'gist_yarg'] + \
                         ['Blues', 'BuPu', 'PuBu', 'PuBuGn', 'BuGn'] + \
                         ['bone', 'gray', 'afmhot', 'hot']
        cmp_colorful = ['CMRmap',
                        'gist_stern',
                        'gnuplot',
                        'gnuplot2',
                        'terrain'] + ['jet',
                                      'bwr',
                                      'coolwarm',
                                      ] + ['Spectral',
                                           'seismic',
                                           'BrBG',
                                           ] + ['rainbow',
                                                'nipy_spectral',
                                                ]

        # set default colormap options & add them to the kwargs
        if colormap is None:
            kwargs['colormap'] = cmp_def + ['--- monochrome ---'] + \
                cmp_monochrome + ['---  colorful  ---'] + cmp_colorful
            # kwargs['colormap'] = ['viridis'] + \
            #    sorted(m for m in plt.cm.datad if not m.endswith("_r"))
        elif isinstance(colormap, str):
            # fix cmap if only one given
            if colormap == 'mono':
                kwargs['colormap'] = cmp_monochrome
            elif colormap == 'color':
                kwargs['colormap'] = cmp_def + cmp_colorful
            else:
                kwargs['colormap'] = fixed(colormap)

        kwargs['figsize'] = fixed(figsize)
        if self.plottype == 'projection':
            self.views = [
                '3D (Sagittal) Projection plane',
                'Coronal slice projection',
                'Axial slice projection']
            self.directions = ['Axial_angle', 'N_u', 'N_v']
        else:
            self.views = ['Sagittal', 'Coronal', 'Axial']
            self.directions = ['x', 'y', 'z']
        self._default_plotter(clim, **kwargs)

    def _default_plotter(self, clim=None, mask_background=False, **kwargs):
        """
        Plot three orthogonal views.
        This is called by nifti_plotter, you shouldn't call it directly.
        """
        data_array = self.data

        if not ((data_array.ndim == 3) or (data_array.ndim == 4)):
            raise ValueError('Input image should be 3D or 4D')

        # mask the background
        if mask_background:
            # TODO: add the ability to pass 'mne' to use a default brain mask
            # TODO: split this out into a different function
            if data_array.ndim == 3:
                labels, n_labels = scipy.ndimage.measurements.label(
                    (np.round(data_array) == 0))
            else:  # 4D
                labels, n_labels = scipy.ndimage.measurements.label(
                    (np.round(data_array).max(axis=3) == 0)
                )

            mask_labels = [lab for lab in range(1, n_labels + 1)
                           if (np.any(labels[[0, -1], :, :] == lab) |
                               np.any(labels[:, [0, -1], :] == lab) |
                               np.any(labels[:, :, [0, -1]] == lab))]

            if data_array.ndim == 3:
                data_array = np.ma.masked_where(
                    np.isin(labels, mask_labels), data_array)
            else:
                data_array = np.ma.masked_where(
                    np.broadcast_to(
                        np.isin(labels, mask_labels)[:, :, :, np.newaxis],
                        data_array.shape
                    ),
                    data_array
                )

        # init sliders for the various dimensions
        for dim, label in enumerate(['x', 'y', 'z']):
            if label not in kwargs.keys():
                kwargs[label] = IntSlider(
                    value=(data_array.shape[dim] - 1) / 2,
                    min=0, max=data_array.shape[dim] - 1,
                    description=self.directions[dim],
                    continuous_update=True
                )

        if (data_array.ndim == 3) or (data_array.shape[3] == 1):
            kwargs['t'] = fixed(0)  # time is fixed
        else:  # 4D
            if self.plottype == 'image':
                desc = 'time'
            elif self.plottype == 'projection':
                desc = 'Azim_angle'
            else:
                desc = 't'
            kwargs['t'] = IntSlider(
                value=data_array.shape[3] // 2,
                min=0,
                max=data_array.shape[3] - 1,
                description=desc,
                continuous_update=True)

        if clim is None:
            clim = (0, data_array.max())
        kwargs['clim'] = FloatRangeSlider(
            value=clim,
            min=0,
            max=clim[1],
            step=0.05,
            description='Contrast',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        interact(self._plot_slices, data=fixed(data_array), **kwargs)

    def _plot_slices(self, data, x, y, z, t, clim,
                     colormap='viridis', figsize=(15, 5)):
        """
        Plot x,y,z slices.
        This function is called by _default_plotter
        """

        if self.image_handles is None:
            self._init_figure(data, colormap, figsize, clim)
        coords = [x, y, z]

        for i, ax in enumerate(self.fig.axes):
            ax.set_title(self.views[i])
        for ii, imh in enumerate(self.image_handles):
            slice_obj = 3 * [slice(None)]
            if data.ndim == 4:
                slice_obj += [t]
            slice_obj[ii] = coords[ii]

            # update the image
            imh.set_data(np.flipud(np.rot90(data[slice_obj], k=1)))
            imh.set_clim(clim)

            # draw guides to show selected coordinates
            guide_positions = [val for jj, val in enumerate(coords)
                               if jj != ii]
            imh.axes.lines[0].set_xdata(2 * [guide_positions[0]])
            imh.axes.lines[1].set_ydata(2 * [guide_positions[1]])

            imh.set_cmap(colormap)

        return self.fig

    def _init_figure(self, data, colormap, figsize, clim):
        # init an empty list
        self.image_handles = []
        # open the figure
        self.fig = plt.figure(figsize=figsize)
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        axes = [ax1, ax2, ax3]

        for ii, ax in enumerate(axes):
            ax.set_facecolor('black')
            ax.tick_params(
                axis='both', which='both', bottom='off', top='off',
                labelbottom='off', right='off', left='off', labelleft='off'
            )
            # fix the axis limits
            axis_limits = [limit for jj, limit in enumerate(data.shape[:3])
                           if jj != ii]
            ax.set_xlim(0, axis_limits[0])
            ax.set_ylim(0, axis_limits[1])

            img = np.zeros(axis_limits[::-1])
            # img[1] = data_max
            im = ax.imshow(img, cmap=colormap,
                           vmin=0.0, vmax=clim)
            # add "cross hair"
            ax.axvline(x=0, color='gray', alpha=0.8)
            ax.axhline(y=0, color='gray', alpha=0.8)
            # append to image handles
            self.image_handles.append(im)


class MultipleVolumes():
    def __init__(
            self,
            volumes,
            axis=0,
            shrink=256,
            rotate=90,
            subsample_slices=None,
            scales=None,
            open_browser=None):
        self.volumes = volumes
        self._axis = axis
        self._shrink = shrink
        self._rotate = rotate
        self._subsample_slices = subsample_slices
        self._scales = scales
        self._open_browser = open_browser
        self._progress_bar = ProgressBar(
            height='6',
            width='100',
            background_color=C.LIGHT_GRAY,
            foreground_color=C.GRAY)

    def get_data(self, volume_index):
        volume = self.volumes[volume_index]
        if isinstance(volume, np.ndarray):
            return volume
        else:
            # Volume type
            return volume.data

    def get_shape(self, volume_index):
        return self.volumes[volume_index].shape

    def export_image(
            self,
            volume_index,
            slice_index,
            axis=0,
            normalise=True,
            scale=None,
            shrink=None,
            rotate=0,
            global_scale=True):
        # FIXME: handle 4D, 5D, ..

        M = self.get_data(volume_index).max()
        m = self.get_data(volume_index).min()

        if axis == 0:
            a = np.float32(self.get_data(volume_index)[slice_index, :, :].reshape(
                (self.get_shape(volume_index)[1], self.get_shape(volume_index)[2])))
        elif axis == 1:
            a = np.float32(self.get_data(volume_index)[:, slice_index, :].reshape(
                (self.get_shape(volume_index)[0], self.get_shape(volume_index)[2])))
        else:
            a = np.float32(self.get_data(volume_index)[:, :, slice_index].reshape(
                (self.get_shape(volume_index)[0], self.get_shape(volume_index)[1])))

        if m >= 0:
            if normalise:
                if not global_scale:
                    if scale is None:
                        a = a * 255 / (a.max() + 1e-9)
                    else:
                        a = a * scale * 255 / (a.max() + 1e-9)
                else:
                    if scale is None:
                        a = a * 255 / (M + 1e-9)
                    else:
                        a = a * scale * 255 / (M + 1e-9)
            im = Image.fromarray(a).convert("RGB")
        else:
            if normalise:
                if not global_scale:
                    if scale is None:
                        a = a * 512 / (a.max() - a.min() + 1e-9)
                    else:
                        a = a * scale * 512 / (a.max() - a.min() + 1e-9)
                else:
                    if scale is None:
                        a = a * 512 / (M - m + 1e-9)
                    else:
                        a = a * scale * 512 / (M - m + 1e-9)
            blue = a
            red = a.copy()
            red[np.where(red >= 0)] = 0
            red = - red
            blue[np.where(blue < 0)] = 0
            green = np.zeros(red.shape)
            rgb = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.uint8)
            rgb[:, :, 0] = red
            rgb[:, :, 1] = green
            rgb[:, :, 2] = blue
            im = Image.fromarray(rgb, mode='RGB')
        if shrink is not None:
            # scale in order to make the largest dimension equal to 'shrink'
            shrink = int(shrink)
            (h, w) = im.size
            # FIXME: downsample the volume (with the GPU, if available) before
            # converting to images, it will save a lot of time, conversion to
            # Image and to RGB is slow
            if h > shrink or w > shrink:
                if h > w:
                    im = im.resize((shrink, int(shrink * w / (1.0 * h))))
                else:
                    im = im.resize((int(shrink * h / (1.0 * w)), shrink))
        if rotate is not None:
            im = im.rotate(rotate)
        return im

    # display
    def display_in_browser(
            self,
            axis=None,
            shrink=None,
            rotate=None,
            subsample_slices=None,
            scales=None):
        self.display(
            axis,
            shrink,
            rotate,
            subsample_slices,
            scales,
            open_browser=True)

    def display(
            self,
            axis=None,
            shrink=None,
            rotate=None,
            subsample_slices=None,
            scales=None,
            open_browser=None):
        if axis is None:
            axis = self._axis
        if shrink is None:
            shrink = self._shrink
        if rotate is None:
            rotate = self._rotate
        if subsample_slices is None:
            subsample_slices = self._subsample_slices
        if subsample_slices is None:
            subsample_slices = 1
        if scales is None:
            scales = self._scales
        if open_browser is None:
            open_browser = self._open_browser
        if open_browser is None:
            open_browser = False
        D = DisplayNode()
        images = []
        n = 0

        N_slices = 0
        for j in range(len(self.volumes)):
            N_slices += self.get_shape(j)[axis]
        self._progress_bar = ProgressBar(
            height='6',
            width='100',
            background_color=C.LIGHT_GRAY,
            foreground_color=C.GRAY)

        for j in range(len(self.volumes)):
            if scales is None:
                scale = 255 / (self.get_data(j).max() + 1e-9)
            else:
                scale = scales[j] * 255 / (self.get_data(j).max() + 1e-9)
            images_inner = []
            for i in range(0, self.get_shape(j)[axis], subsample_slices):
                im = self.export_image(
                    j,
                    i,
                    axis,
                    normalise=True,
                    scale=scale,
                    shrink=shrink,
                    rotate=rotate)
                images_inner.append(im)
                n += 1
                self._progress_bar.set_percentage(
                    n * 100.0 / N_slices * subsample_slices)
            images.append(images_inner)
        return D.display('tipix', images, open_browser)

    def _repr_html_(self):
        return self.display()._repr_html_()


class MultipleVolumesNiftyPy():
    def __init__(self, volumes, axis=0, open_browser=None):
        self.volumes = volumes
        self._axis = axis
        self._open_browser = open_browser
        self._progress_bar = ProgressBar(
            height='6px',
            width='100%%',
            background_color=C.LIGHT_GRAY,
            foreground_color=C.GRAY)

    def get_data(self, volume_index):
        volume = self.volumes[volume_index]
        if isinstance(volume, np.ndarray):
            return volume
        else:
            # Image3D
            return volume.data

    def get_shape(self, volume_index):
        return self.volumes[volume_index].shape

    def _resample_volume(self):
        pass

        # display

    def display_in_browser(self, axis=None, max_size=200):
        self.display(axis, max_size, open_browser=True)

    def display(self, axis=None, max_size=256, open_browser=None):
        if axis is None:
            axis = self._axis
        if open_browser is None:
            open_browser = self._open_browser
        if open_browser is None:
            open_browser = False
        D = DisplayNode()

        self._progress_bar = ProgressBar(
            height='6px',
            width='100%%',
            background_color=C.LIGHT_GRAY,
            foreground_color=C.GRAY)

        volume = self.volumes[0]  # FIXME: optionally use other grid
        # make grid of regularly-spaced points
        box_min = volume.get_world_grid_min()
        box_max = volume.get_world_grid_max()
        span = box_max - box_min
        max_span = span.max()
        n_points = np.uint32(span / max_span * max_size)
        grid = volume.get_world_grid(n_points)
        n = 0
        images = []
        for index in range(len(self.volumes)):
            volume = self.volumes[index]
            resampled = volume.compute_resample_on_grid(grid).data
            self._resampled = resampled
            sequence = []
            for slice_index in range(n_points[axis]):
                if axis == 0:
                    a = np.float32(resampled[slice_index, :, :].reshape(
                        (resampled.shape[1], resampled.shape[2])))
                elif axis == 1:
                    a = np.float32(resampled[:, slice_index, :].reshape(
                        (resampled.shape[0], resampled.shape[2])))
                else:
                    a = np.float32(resampled[:, :, slice_index].reshape(
                        (resampled.shape[0], resampled.shape[1])))
                lookup_table = volume.get_lookup_table()
                im = self.__array_to_im(a, lookup_table)
                sequence.append(im.rotate(90))  # FIXME: make optional
                n += 1
                self._progress_bar.set_percentage(
                    n * 100.0 / (len(self.volumes) * max_size))
            images.append(sequence)
        if len(images) == 1:
            return D.display('tipix', images[0], open_browser)
        else:
            return D.display('tipix', images, open_browser)

    def __array_to_im(self, a, lookup_table):
        if lookup_table is not None:
            red, green, blue, alpha = lookup_table.convert_ndarray_to_rgba(a)
            rgb = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.uint8)
            rgb[:, :, 0] = red
            rgb[:, :, 1] = green
            rgb[:, :, 2] = blue
            im = Image.fromarray(rgb, mode='RGB')
        else:
            im = Image.fromarray(a).convert("RGB")
        return im

    def _repr_html_(self):
        return self.display()._repr_html_()


def deg_to_rad(deg):
    return deg * np.pi / 180.0


def rad_to_deg(rad):
    return rad * 180.0 / np.pi


try:
    from ...NiftyPy.NiftyRec import SPECT_project_parallelholes as projection
except BaseException:
    has_NiftyPy = False
    print("Please install NiftyPy")
else:
    has_NiftyPy = True


class SPECT_Projection():
    """SPECT projection object. """

    def __init__(self, data):
        self.data = data

    def get_data(self):
        """Returns the raw projection data (note that is can be accessed also as self.data ). """
        return self.data

    def save_to_file(self, filename):
        h5f = _h5py.File(filename, 'w')
        h5f.create_dataset('data', data=self.data)
        # h5f.create_dataset('size_x', data=size_x)
        # h5f.create_dataset('size_y', data=size_y)
        h5f.close()

    def get_integral(self):
        return self.data.sum()

    def to_image(self, data, index=0, scale=None, absolute_scale=False):
        from PIL import Image
        a = np.float32(data[:, :, index].reshape(
            (data.shape[0], data.shape[1])))
        if scale is None:
            a = 255.0 * (a) / (a.max() + 1e-12)
        else:
            if absolute_scale:
                a = scale * (a)
            else:
                a = scale * 255.0 * (a) / (a.max() + 1e-12)
        return Image.fromarray(a).convert("RGB")

    def display_in_browser(
            self,
            axial=True,
            azimuthal=False,
            index=0,
            scale=None):
        self.display(
            axial=axial,
            azimuthal=azimuthal,
            index=index,
            scale=scale,
            open_browser=True)

    def display(self, scale=None, open_browser=False):
        data = self.data
        d = DisplayNode()
        images = []
        progress_bar = ProgressBar(
            height='6px',
            width='100%%',
            background_color=C.LIGHT_GRAY,
            foreground_color=C.GRAY)
        if scale is not None:
            scale = scale * 255.0 / (data.max() + 1e-12)
        else:
            scale = 255.0 / (data.max() + 1e-12)
        N_projections = self.data.shape[2]
        N_x = self.data.shape[0]
        N_y = self.data.shape[1]
        print(
            "SPECT Projection   [N_projections: %d   N_x: %d   N_y: %d]" %
            (N_projections, N_x, N_y))
        for i in range(N_projections):
            images.append(
                self.to_image(
                    data,
                    i,
                    scale=scale,
                    absolute_scale=True))
            progress_bar.set_percentage(i * 100.0 / N_projections)
        progress_bar.set_percentage(100.0)
        return d.display('tipix', images, open_browser)

    def _repr_html_(self):
        return self.display()._repr_html_()


class VolumeRenderer():
    def __init__(
            self,
            volume,
            theta_min_deg=0.0,
            theta_max_deg=360.0,
            n_positions=180,
            truncate_negative=False,
            psf=None,
            attenuation=None,
            scale=1.0):
        self.volume = volume
        self.psf = psf
        self.attenuation = attenuation
        self.use_gpu = is_gpu_enabled()
        self.theta_min_deg = theta_min_deg
        self.theta_max_deg = theta_max_deg
        self.n_positions = n_positions
        self.truncate_negative = truncate_negative
        self.scale = scale

    def __make_cameras(self, axis, direction):
        # self.cameras = np.float32(np.linspace(deg_to_rad(self.theta_min_deg),deg_to_rad(self.theta_max_deg),self.n_positions).reshape((self.n_positions,1)))
        self.cameras = np.zeros((self.n_positions, 3), dtype=np.float32)
        self.cameras[:, 0] = self.cameras[:, 0] + deg_to_rad(axis[0])
        self.cameras[:, 1] = self.cameras[:, 0] + deg_to_rad(axis[1])
        self.cameras[:, 2] = self.cameras[:, 0] + deg_to_rad(axis[2])
        self.cameras[:, direction] = np.linspace(deg_to_rad(
            self.theta_min_deg), deg_to_rad(self.theta_max_deg), self.n_positions)

    def render(self, axis=(90, 0, 0), direction=0, max_n_points=None):
        if hasattr(
                self.volume,
                'compute_resample_on_grid'):  # i.e. if it is a Image3
            volume = self.volume.copy()
            # make grid of regularly-spaced points
            if max_n_points is None:
                max_n_points = 256
            box_min = volume.get_world_grid_min()
            box_max = volume.get_world_grid_max()
            span = box_max - box_min
            max_span = span.max()
            n_points = np.uint32(span / max_span * max_n_points)
            grid = volume.get_world_grid(n_points)
            volume.compute_resample_on_grid(grid)
            volume = np.float32(volume.data)
        else:
            volume = self.volume
        self.__make_cameras(axis, direction)
        if has_NiftyPy:
            proj_data = projection(
                volume,
                self.cameras,
                self.attenuation,
                self.psf,
                0.0,
                0.0,
                self.use_gpu,
                self.truncate_negative)
        else:
            raise InstallationError(
                "NiftyPy not installed, please install to execute render(). ")
        self.__proj = SPECT_Projection(proj_data)
        # FIXME: memoize projection (use new style objects - properties)
        return self.__proj

    def display(self):
        r = self.render()
        return r.display(scale=self.scale)

    def _repr_html_(self):
        return self.display()


H_graph = """
<div id="__id__"></div>
<style type="text/css">
path.link {
  fill: none;
  stroke: #666;
  stroke-width: 1.5px;
}

marker#t0 {
  fill: green;
}

path.link.t0 {
  stroke: green;
}

path.link.t2 {
  stroke-dasharray: 0,2 1;
}

circle {
  fill: #ccc;
  stroke: #333;
  stroke-width: 1.5px;
}

text {
  font: 10px sans-serif;
  pointer-events: none;
}

text.shadow {
  stroke: #fff;
  stroke-width: 3px;
  stroke-opacity: .8;
}
</style>
"""

J_graph = """
<script type="text/Javascript">
require.config({paths: {d3: "http://d3js.org/d3.v3.min"}});
require(["d3"], function(d3) {

var width = 800;
var height = 500;

var graph = __graph_data__;

var nodes = graph['nodes'];
var links = graph['links'];

var nodes2 = {};
var links2 = [];

for (var i=0; i<nodes.length; i++) {
    node = nodes[i];
    nodes2[node['name']] = node;
};

for (var i=0; i<links.length; i++) {
    links2[i] = {'source':nodes2[links[i]['source']], 'target':nodes2[links[i]['target']], 'type':links[i]['type'],};
};


var force = d3.layout.force()
    .nodes(d3.values(nodes2))
    .links(links2)
    .size([width, height])
    .linkDistance(60)
    .charge(-300)
    .on("tick", tick)
    .start();

var svg = d3.select("#__id__").append("svg:svg")
    .attr("width", width)
    .attr("height", height);

// Per-type markers, as they don't inherit styles.
svg.append("svg:defs").selectAll("marker")
    .data(["t0", "t1", "t2"])
  .enter().append("svg:marker")
    .attr("id", String)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 20)
    .attr("refY", -1.5)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
  .append("svg:path")
    .attr("d", "M0,-5L10,0L0,5");

var path = svg.append("svg:g").selectAll("path")
    .data(force.links())
  .enter().append("svg:path")
    .attr("class", function(d) { return "link " + d.type; })
    .attr("marker-end", function(d) { return "url(#" + d.type + ")"; });

colors  = ['#fff','#ccc','#ee4','#dddddd','#eeeeee','#ffffc0'];
strokes = ['#333','#333','#333','#dddddd','#dddddd','#dddddd'];

var circle = svg.append("svg:g").selectAll("circle")
    .data(force.nodes())
  .enter().append("svg:circle")
    .attr("r", 10)
    .call(force.drag)
    .style('fill', function(d){return colors[d.type];})
    .style('stroke', function(d){return strokes[d.type];});

var text = svg.append("svg:g").selectAll("g")
    .data(force.nodes())
  .enter().append("svg:g");

// A copy of the text with a thick white stroke for legibility.
text.append("svg:text")
    .attr("x", 12)
    .attr("y", ".31em")
    .attr("class", "shadow")
    .text(function(d) { return d.name; });

text.append("svg:text")
    .attr("x", 12)
    .attr("y", ".31em")
    .text(function(d) { return d.name; });


// Use elliptical arc path segments to doubly-encode directionality.
function tick() {
  path.attr("d", function(d) {
    var dx = d.target.x - d.source.x,
        dy = d.target.y - d.source.y,
        dr = Math.sqrt(dx * dx + dy * dy);
    return "M" + d.source.x + "," + d.source.y + "A" + dr + "," + dr + " 0 0,1 " + d.target.x + "," + d.target.y;
  });

  circle.attr("transform", function(d) {
    return "translate(" + d.x + "," + d.y + ")";
  });

  text.attr("transform", function(d) {
    return "translate(" + d.x + "," + d.y + ")";
  });
}

});
</script>
"""

import json


class Graph():
    def __init__(self, graph):
        self.set_graph(graph)

    def set_graph(self, g):
        self.graph = g

    def get_html(self):
        div_id = "graph_" + str(_uuid.uuid4())
        H = H_graph.replace("__id__", div_id)
        J = J_graph.replace("__id__", div_id)
        J = J.replace("__graph_data__", json.dumps(self.graph))
        return H + J

    def _repr_html_(self):
        graph = HTML(self.get_html())
        return graph._repr_html_()

        # ipy_table


# FIXME: perhaps move this somewhere else

try:
    import ipy_table
    has_ipy_table = True
except BaseException:
    print("Please install ipy_table (e.g. 'easy_install ipy_table') to enable ipython notebook tables. ")
    ipy_table = None
    has_ipy_table = False
# svg_write
# FIXME: perhaps move this somewhere else
try:
    import svgwrite
    has_svgwrite = True
except BaseException:
    print("Please install svgwrite (e.g. 'easy_install svgwrite') to enable svg visualisations. ")
    svgwrite = None
    has_svgwrite = False
