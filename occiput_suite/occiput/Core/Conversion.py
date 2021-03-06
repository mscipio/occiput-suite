# occiput
# Stefano Pedemonte
# Harvard University, Martinos Center for Biomedical Imaging
# Dec. 2014, Boston, MA
# April. 2014, Boston, MA

from __future__ import absolute_import, print_function
import warnings as __warnings
import nibabel as __nibabel
import occiput_suite as __occiput_suite

with __warnings.catch_warnings():
    __warnings.simplefilter("ignore")
    from nipy.io.nifti_ref import nifti2nipy as nifti_to_nipy


def nibabel_to_occiput(nib):
    ndim = len(nib.shape)
    if ndim == 3:
        im = __occiput_suite.occiput.Core.Core.Image3D(
            data=nib.get_data(), affine=nib.affine,
            space="world", header=nib.header)
    else:
        im = __occiput_suite.occiput.Core.Core.ImageND(
            data=nib.get_data(), affine=nib.affine,
            space="world", header=nib.header)
    return im


def nifti_to_occiput(nif):
    nip = nifti_to_nipy(nif)
    return nibabel_to_occiput(nip)


def occiput_to_nifti(occ):
    nii = __nibabel.nifti1.Nifti1Image(occ.data, occ.affine.data, occ.header)
    return nii


def occiput_from_array(array):
    if array.ndim == 3:
        im = __occiput_suite.occiput.Core.Core.Image3D(
            data=array, space="world")
    else:
        raise("Currently only conversion of 3D arrays is supported. ")
    return im
