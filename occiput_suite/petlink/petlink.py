# petlink - Decode and encode PETlink streams.
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 
# Martinos Center for Biomedical Imaging, Harvard University/MGH, Boston
# Dec. 2013, Boston

from __future__ import absolute_import, print_function

__all__ = ['PET_Interface_Petlink32']

from ..simplewrap import find_c_library, localpath, load_c_library, call_c_function
from numpy import int32, uint16, uint32, float32


class ErrorInCFunction(Exception):
    def __init__(self, msg, status, function_name):
        self.msg = str(msg)
        self.status = status
        self.function_name = function_name
        if self.status == status_io_error():
            self.status_msg = "IO Error"
        elif self.status == status_decode_error():
            self.status_msg = "Error Decoding file content"
        elif self.status == status_initialisation_error():
            self.status_msg = "Error with the initialisation of the C library"
        elif self.status == status_parameter_error():
            self.status_msg = "One or more of the specified parameters are not right"
        elif self.status == status_unhandled_error():
            self.status_msg = "Unhandled error, likely a bug. "
        else:
            self.status_msg = "Unspecified Error"

    def __str__(self):
        return "'%s' returned by the C Function '%s'. %s" % (self.status_msg, self.function_name, self.msg)


# Load library
(found, fullpath, path) = find_c_library("petlink32_c", [localpath(), ])
if found:
    petlink32_c = load_c_library(fullpath)
else:
    raise  RuntimeError("Library petlink32_c Not Found")


# Utility functions 


def status_success():
    """Returns the value returned by the function calls to the library in case of success. """
    r = call_c_function(petlink32_c.status_success, [{'name': 'return_value', 'type': 'int', 'value': None}])
    return r.return_value


def status_io_error():
    """Returns the integer value returned by the function calls to the library in case of IO error. """
    r = call_c_function(petlink32_c.status_io_error, [{'name': 'return_value', 'type': 'int', 'value': None}])
    return r.return_value


def status_decode_error():
    """Returns the value returned by the function calls to the library in case of error decoding a file. """
    r = call_c_function(petlink32_c.status_decode_error, [{'name': 'return_value', 'type': 'int', 'value': None}])
    return r.return_value


def status_initialisation_error():
    """Returns the value returned by the function calls to the library in case of initialisation error. """
    r = call_c_function(petlink32_c.status_initialisation_error,
                        [{'name': 'return_value', 'type': 'int', 'value': None}])
    return r.return_value


def status_parameter_error():
    """Returns the value returned by the function calls to the library in case of parameter error. """
    r = call_c_function(petlink32_c.status_parameter_error, [{'name': 'return_value', 'type': 'int', 'value': None}])
    return r.return_value


def status_unhandled_error():
    """Returns the value returned by the function calls to the library in case of unhandled error. """
    r = call_c_function(petlink32_c.status_unhandled_error, [{'name': 'return_value', 'type': 'int', 'value': None}])
    return r.return_value


# Create interface to the C functions:
def test_library_petlink32_c():
    """Test whether the C library petlink32_c responds. """
    number = 101  # just a number
    descriptor = [{'name': 'input', 'type': 'int', 'value': number},
                  {'name': 'output', 'type': 'int', 'value': None}, ]
    r = call_c_function(petlink32_c.echo, descriptor)
    return r.output == number


def petlink32_info(filename, n_packets):
    """Extracts summary information from a listmode binary file. """
    descriptor = [{'name': 'filename', 'type': 'string', 'value': filename, 'size': len(filename)},
                  {'name': 'n_packets', 'type': 'int', 'value': n_packets},
                  {'name': 'n_prompts', 'type': 'int', 'value': None},
                  {'name': 'n_delayed', 'type': 'int', 'value': None},
                  {'name': 'n_tags', 'type': 'int', 'value': None},
                  {'name': 'n_time', 'type': 'int', 'value': None},
                  {'name': 'n_motion', 'type': 'int', 'value': None},
                  {'name': 'n_monitoring', 'type': 'int', 'value': None},
                  {'name': 'n_control', 'type': 'int', 'value': None}, ]
    r = call_c_function(petlink32_c.petlink32_info, descriptor)
    if not r.status == status_success():
        raise ErrorInCFunction("The execution of petlink32_info was unsuccesful.", r.status,
                               'petlink32_c.petlink32_info')
    return r.dictionary


def petlink32_bin_addresses(filename, n_packets):
    """Extract list of bin indexes from listmode data. """
    descriptor = [{'name': 'filename', 'type': 'string', 'value': filename, 'size': len(filename)},
                  {'name': 'n_packets', 'type': 'int', 'value': n_packets},
                  {'name': 'bin_addresses', 'type': 'array', 'value': None, 'dtype': int32, 'size': (1, n_packets)},
                  {'name': 'n_prompts', 'type': 'int', 'value': None},
                  {'name': 'n_delayed', 'type': 'int', 'value': None},
                  {'name': 'n_tags', 'type': 'int', 'value': None},
                  {'name': 'n_time', 'type': 'int', 'value': None},
                  {'name': 'n_motion', 'type': 'int', 'value': None},
                  {'name': 'n_monitoring', 'type': 'int', 'value': None},
                  {'name': 'n_control', 'type': 'int', 'value': None}, ]
    r = call_c_function(petlink32_c.petlink32_bin_addresses, descriptor)
    if not r.status == status_success():
        raise ErrorInCFunction("The execution of petlink32_bin_addresses was unsuccesful.", r.status,
                               'petlink32_c.petlink32_bin_addresses')
    return r.dictionary


def petlink32_to_static_sinogram(filename, n_packets, n_radial_bins, n_angles, n_sinograms):
    """Make static sinogram from listmode data. """
    descriptor = [{'name': 'filename', 'type': 'string', 'value': filename, 'size': len(filename)},
                  {'name': 'n_packets', 'type': 'long', 'value': n_packets},
                  {'name': 'n_radial_bins', 'type': 'int', 'value': n_radial_bins},
                  {'name': 'n_angles', 'type': 'int', 'value': n_angles},
                  {'name': 'n_sinograms', 'type': 'int', 'value': n_sinograms},
                  {'name': 'sinogram', 'type': 'array', 'value': None, 'dtype': int32,
                   'size': (n_radial_bins, n_angles, n_sinograms)},
                  {'name': 'packets_unknown', 'type': 'int', 'value': None},
                  {'name': 'packets_prompt', 'type': 'int', 'value': None},
                  {'name': 'packets_delayed', 'type': 'int', 'value': None},
                  {'name': 'packets_elapsedtime', 'type': 'int', 'value': None},
                  {'name': 'packets_deadtime', 'type': 'int', 'value': None},
                  {'name': 'packets_motion', 'type': 'int', 'value': None},
                  {'name': 'packets_gating', 'type': 'int', 'value': None},
                  {'name': 'packets_tracking', 'type': 'int', 'value': None},
                  {'name': 'packets_control', 'type': 'int', 'value': None}, ]
    r = call_c_function(petlink32_c.petlink32_to_static_sinogram, descriptor)
    if not r.status == status_success():
        raise ErrorInCFunction("The execution of 'petlink32_to_static_sinogram' was unsuccesful.", r.status,
                               'petlink32_c.petlink32_to_static_sinogram')
    return r.dictionary


def petlink32_to_dynamic_projection(filename, n_packets, n_radial_bins, n_angles, n_sinograms, time_bins, n_axial,
                                    n_azimuthal, angular_step_axial, angular_step_azimuthal, size_u, size_v, n_u, n_v):
    """Make dynamic compressed projection from list-mode data. """
    descriptor = [{'name': 'filename', 'type': 'string', 'value': filename, 'size': len(filename)},
                  {'name': 'n_packets', 'type': 'long', 'value': n_packets},
                  {'name': 'n_radial_bins', 'type': 'uint', 'value': n_radial_bins},
                  {'name': 'n_angles', 'type': 'uint', 'value': n_angles},
                  {'name': 'n_sinograms', 'type': 'uint', 'value': n_sinograms},
                  {'name': 'n_time_bins', 'type': 'uint', 'value': len(time_bins)},
                  {'name': 'time_bins', 'type': 'array', 'value': int32(time_bins)},
                  {'name': 'n_axial', 'type': 'uint', 'value': n_axial},
                  {'name': 'n_azimuthal', 'type': 'uint', 'value': n_azimuthal},
                  {'name': 'angular_step_axial', 'type': 'float', 'value': angular_step_axial},
                  {'name': 'angular_step_azimuthal', 'type': 'float', 'value': angular_step_azimuthal},
                  {'name': 'size_u', 'type': 'float', 'value': size_u},
                  {'name': 'size_v', 'type': 'float', 'value': size_v},
                  {'name': 'n_u', 'type': 'uint', 'value': n_u},
                  {'name': 'n_v', 'type': 'uint', 'value': n_v}, ]
    r = call_c_function(petlink32_c.petlink32_to_dynamic_projection, descriptor)
    if not r.status == status_success():
        raise ErrorInCFunction("The execution of 'petlink32_to_dynamic_compressed_sinogram' was unsuccesful.", r.status,
                               'petlink32_c.petlink32_to_dynamic_compressed_sinogram')
    return r.dictionary


def get_petlink32_stats():
    """Get information about the current petlink32 listmode data. """
    descriptor = [{'name': 'n_packets', 'type': 'long', 'value': None},
                  {'name': 'n_unknown', 'type': 'uint', 'value': None},
                  {'name': 'n_prompt', 'type': 'uint', 'value': None},
                  {'name': 'n_delayed', 'type': 'uint', 'value': None},
                  {'name': 'n_elapsedtime', 'type': 'uint', 'value': None},
                  {'name': 'n_deadtime', 'type': 'uint', 'value': None},
                  {'name': 'n_motion', 'type': 'uint', 'value': None},
                  {'name': 'n_gating', 'type': 'uint', 'value': None},
                  {'name': 'n_tracking', 'type': 'uint', 'value': None},
                  {'name': 'n_control', 'type': 'uint', 'value': None}, ]
    r = call_c_function(petlink32_c.get_petlink32_stats, descriptor)
    if not r.status == status_success():
        raise ErrorInCFunction("The execution of 'get_petlink32_stats' was unsuccesful.", r.status,
                               'petlink32_c.get_petlink32_stats')
    return r.dictionary


def get_dynamic_projection_info():
    """Get information about the current dynamic projection data. """
    descriptor = [{'name': 'N_time_bins', 'type': 'uint', 'value': None},
                  {'name': 'N_counts', 'type': 'long', 'value': None}, ]
    r = call_c_function(petlink32_c.get_dynamic_projection_info, descriptor)
    if not r.status == status_success():
        raise ErrorInCFunction("The execution of 'get_dynamic_projection_info' was unsuccesful.", r.status,
                               'petlink32_c.get_dynamic_projection_info')
    return r.dictionary


def get_static_projection_info(time_bin):
    """Get information about a time bin of the current dynamic projection data. """
    descriptor = [{'name': 'time_bin', 'type': 'uint', 'value': time_bin},
                  {'name': 'time_start', 'type': 'uint', 'value': None},
                  {'name': 'time_end', 'type': 'uint', 'value': None},
                  {'name': 'N_counts', 'type': 'uint', 'value': None},
                  {'name': 'N_axial', 'type': 'uint', 'value': None},
                  {'name': 'N_azimuthal', 'type': 'uint', 'value': None},
                  {'name': 'angular_step_axial', 'type': 'float', 'value': None},
                  {'name': 'angular_step_azimuthal', 'type': 'float', 'value': None},
                  {'name': 'size_u', 'type': 'float', 'value': None},
                  {'name': 'size_v', 'type': 'float', 'value': None},
                  {'name': 'N_u', 'type': 'uint', 'value': None},
                  {'name': 'N_v', 'type': 'uint', 'value': None}, ]
    r = call_c_function(petlink32_c.get_static_projection_info, descriptor)
    if not r.status == status_success():
        raise ErrorInCFunction("The execution of 'get_static_projection_info' was unsuccesful.", r.status,
                               'petlink32_c.get_static_projection_info')
    return r.dictionary


def get_static_projection(time_bin):
    """Get the static projection corresponding to a given time bin of the current dynamic projection data. """
    info = get_static_projection_info(time_bin)
    N_counts = info['N_counts']
    N_axial = info['N_axial']
    N_azimuthal = info['N_azimuthal']
    descriptor = [{'name': 'time_bin', 'type': 'uint', 'value': time_bin},
                  {'name': 'time_start', 'type': 'uint', 'value': None},
                  {'name': 'time_end', 'type': 'uint', 'value': None},
                  {'name': 'N_counts', 'type': 'uint', 'value': None},
                  {'name': 'N_axial', 'type': 'uint', 'value': None},
                  {'name': 'N_azimuthal', 'type': 'uint', 'value': None},
                  {'name': 'angular_step_axial', 'type': 'float', 'value': None},
                  {'name': 'angular_step_azimuthal', 'type': 'float', 'value': None},
                  {'name': 'size_u', 'type': 'float', 'value': None},
                  {'name': 'size_v', 'type': 'float', 'value': None},
                  {'name': 'N_u', 'type': 'uint', 'value': None},
                  {'name': 'N_v', 'type': 'uint', 'value': None},
                  {'name': 'offsets', 'type': 'array', 'value': None, 'dtype': uint32, 'size': (N_axial, N_azimuthal)},
                  {'name': 'counts', 'type': 'array', 'value': None, 'dtype': float32, 'size': N_counts},
                  {'name': 'locations', 'type': 'array', 'value': None, 'dtype': uint16, 'size': (N_counts, 3)}, ]
    r = call_c_function(petlink32_c.get_static_projection, descriptor)
    if not r.status == status_success():
        raise ErrorInCFunction("The execution of 'get_static_projection' was unsuccesful.", r.status,
                               'petlink32_c.get_static_projection')
    return r.dictionary


def uncompress_static_projection(offsets, counts, locations, N_u, N_v):
    """Get the static projection corresponding to a given time bin of the current dynamic projection data. """
    N_counts = counts.size
    N_axial = offsets.shape[0]
    N_azimuthal = offsets.shape[1]
    descriptor = [{'name': 'N_counts', 'type': 'uint', 'value': N_counts},
                  {'name': 'N_axial', 'type': 'uint', 'value': N_axial},
                  {'name': 'N_azimuthal', 'type': 'uint', 'value': N_azimuthal},
                  {'name': 'N_u', 'type': 'uint', 'value': N_u},
                  {'name': 'N_v', 'type': 'uint', 'value': N_v},
                  {'name': 'offsets', 'type': 'array', 'value': offsets},
                  {'name': 'counts', 'type': 'array', 'value': counts},
                  {'name': 'locations', 'type': 'array', 'value': locations},
                  {'name': 'projection', 'type': 'array', 'value': None, 'dtype': uint32,
                   'size': (N_axial, N_azimuthal, N_u, N_v)}, ]
    r = call_c_function(petlink32_c.uncompress_static_projection, descriptor)
    if not r.status == status_success():
        raise ErrorInCFunction("The execution of 'uncompress_static_projection' was unsuccesful.", r.status,
                               'petlink32_c.uncompress_static_projection')
    return r.dictionary


def free_memory():
    """Free memory"""
    r = call_c_function(petlink32_c.free_memory, [])
    if not r.status == status_success():
        raise ErrorInCFunction("The execution of 'free_memory' was unsuccesful.", r.status, 'petlink32_c.free_memory')
    return r.dictionary


class PET_Interface_Petlink32:
    def __init__(self):
        self.name = "Generic Scanner"
        self.manufacturer = "Unknown Manufacturer"
        self.version = "0.0"

    def listmode_to_dynamic_measurement(self, listmode_data_filename, n_packets, time_bins, binning, n_radial_bins,
                                        n_angles, n_sinograms):
        n_axial = binning.N_axial
        n_azimuthal = binning.N_azimuthal
        angular_step_axial = binning.angular_step_axial
        angular_step_azimuthal = binning.angular_step_azimuthal
        size_u = binning.size_u
        size_v = binning.size_v
        n_u = binning.N_u
        n_v = binning.N_v
        return petlink32_to_dynamic_projection(listmode_data_filename, n_packets, n_radial_bins, n_angles, n_sinograms,
                                               time_bins, n_axial, n_azimuthal, angular_step_axial,
                                               angular_step_azimuthal, size_u, size_v, n_u, n_v)

    def __del__(self):
        free_memory()
        print("Dynamic PET data structures deallocated. ")
