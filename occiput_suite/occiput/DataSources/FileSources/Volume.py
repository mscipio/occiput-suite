# occiput
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 
# Harvard University, Martinos Center for Biomedical Imaging 
# Boston, MA, USA


import os
import warnings

import dicom
import numpy

try:
    import dcmstack
    dcmstack_available = True
except:
    dcmstack_available = False

from glob import glob

from ...Core.Conversion import nibabel_to_occiput, nifti_to_occiput
from ...Visualization.LookupTable import load_freesurfer_lut_file
from ...Visualization.Visualization import ProgressBar

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # import nipy
    import nibabel

##############################################################################
# We need to override this method of dcmstack class
# in order to allow for float32 datatype for the
# volume array. High emission images in Bq/cc units
# cannot be contrainted in the limited ammisible range
# of int16. Moreover, there's no point in unsing integer
# values for emission data
#-----------------------------------------------------------------------------
def ovverride_dcmstack_get_data(self):
    import numpy as np
    '''Get an array of the voxel values.

    Returns
    -------
    A numpy array filled with values from the DICOM data sets' pixels.

    Raises
    ------
    InvalidStackError
        The stack is incomplete or invalid.
    '''
    # Create a numpy array for storing the voxel data
    stack_shape = self.get_shape()
    stack_shape = tuple(list(stack_shape) + ((5 - len(stack_shape)) * [1]))
    vox_array = np.empty(stack_shape, np.float32)

    # Fill the array with data
    n_vols = 1
    if len(stack_shape) > 3:
        n_vols *= stack_shape[3]
    if len(stack_shape) > 4:
        n_vols *= stack_shape[4]
    files_per_vol = len(self._files_info) / n_vols
    file_shape = self._files_info[0][0].nii_img.get_shape()
    for vec_idx in range(stack_shape[4]):
        for time_idx in range(stack_shape[3]):
            if files_per_vol == 1 and file_shape[2] != 1:
                file_idx = vec_idx * (stack_shape[3]) + time_idx
                vox_array[:, :, :, time_idx, vec_idx] = \
                    self._files_info[file_idx][0].nii_img.get_data()
            else:
                for slice_idx in range(files_per_vol):
                    file_idx = (vec_idx * (stack_shape[3] * stack_shape[2]) +
                                time_idx * (stack_shape[2]) + slice_idx)
                    vox_array[:, :, slice_idx, time_idx, vec_idx] = \
                        self._files_info[file_idx][0].nii_img.get_data()[:, :, 0]
    # Trim unused time/vector dimensions
    if stack_shape[4] == 1:
        vox_array = vox_array[..., 0]
        if stack_shape[3] == 1:
            vox_array = vox_array[..., 0]

    return vox_array

dcmstack.DicomStack.get_data = ovverride_dcmstack_get_data
##############################################################################

def import_nifti(filename):
    # nip = nipy.load_image(filename)
    nip = nibabel.load(filename)
    img = nibabel_to_occiput(nip)
    return img


def import_mask(filename, lookup_table_filename=None):
    # Load file 
    # nip = nipy.load_image(filename)
    nip = nibabel.load(filename)
    occ = nibabel_to_occiput(nip)
    occ.set_mask_flag(1)

    # Load the lookup table. If not specified, try to load from file with the same name as 
    # the mask image file. 
    if lookup_table_filename == None:
        f = []
        f.append(os.path.splitext(filename)[0] + '.lut')
        f.append(os.path.splitext(os.path.splitext(filename)[0])[0] + '.lut')  # This includes .nii.gz files
        for lookup_table_filename in f:
            try:
                lut = load_freesurfer_lut_file(lookup_table_filename)
            except:
                lut = None

    else:
        lut = load_freesurfer_lut_file(lookup_table_filename)
    if lut is not None:
        occ.set_lookup_table(lut)
    return occ


def import_dicom(search_path, extension='IMA'):
    progress_bar = ProgressBar(title='Reading src')
    progress_bar.set_percentage(0.1)
    if (not dcmstack_available):
        progress_bar.set_percentage(100.0)
        raise ("Pleast install dcmstack from https://github.com/moloney/dcmstack/tags")
    else:
        search_string = search_path + '/*.' + extension
        src_paths = glob(search_string)
        stacks = dcmstack.parse_and_stack(src_paths)
        images = []
        for k, key in enumerate(stacks.keys()):
            stack = stacks[key]
            img = nibabel_to_occiput(stack.to_nifti(embed_meta=True))
            images.append(img)
            progress_bar.set_percentage((k + 1) * 100.0 / len(stacks.keys()))
        progress_bar.set_percentage(100.0)
        return images


def import_dicom_series(path, files_start_with=None, files_end_with=None,
                        exclude_files_end_with=('.dat', '.txt', '.py', '.pyc', '.nii', '.gz')):
    """Rudimentary file to load dicom serie from a directory. """
    N = 0
    paths = []
    slices = []
    files = os.listdir(path)
    progress_bar = ProgressBar(title='Reading src')
    progress_bar.set_percentage(0.1)

    for k, file_name in enumerate(files):
        file_valid = True
        if files_start_with is not None:
            if not file_name.startswith(files_start_with):
                file_valid = False
        if files_end_with is not None:
            if not file_name.endswith(files_end_with):
                file_valid = False
        for s in exclude_files_end_with:
            if file_name.endswith(s):
                file_valid = False
        if file_valid:
            # print(file_name)
            full_path = path + os.sep + file_name
            # read moco information from files
            paths.append(full_path)
            f = dicom.read_file(full_path)
            slice = f.pixel_array
            slices.append(slice)
            N += 1
            instance_number = f.get(0x00200013).value
            creation_time = f.get(0x00080013).value
            # print "Instance number:    ",instance_number
            # print "Creation time:      ",creation_time
        progress_bar.set_percentage((k + 1) * 100.0 / len(files))

    progress_bar.set_percentage(100.0)
    array = numpy.zeros((slices[0].shape[0], slices[0].shape[1], N), dtype=numpy.float32)
    for i in range(N):
        slice = numpy.float32(slices[i])  # FIXME: handle other data types
        array[:, :, i] = slice
        # return occiput_from_array(array)
    return array


