import sys 
sys.path.insert(0, '/media/DATA/DOCUMENTI/GITHUB/__OCCIPUT_src/occiput-suite-DEV')
from occiput_suite.simplewrap import find_c_library, load_c_library, localpath, call_c_function, filepath
from numpy import int32, uint32, uint16, float16, float32
import dicom 

__all__ = ['Biograph_mMR_Physiology']


(found,fullpath,path) = find_c_library("physiological_c",[localpath(), filepath(__file__)]) 
physiological_c = load_c_library(fullpath)



###### Exceptions: ########
class ErrorInCFunction(Exception): 
    def __init__(self,msg,status,function_name): 
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
        return "'%s' returned by the C Function '%s'. %s"%(self.status_msg,self.function_name,self.msg)

###### Utilities interface C library: ########
def status_success(): 
    """Returns the value returned by the function calls to the library in case of success. """
    r = call_c_function( physiological_c.status_success, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value

def status_io_error(): 
    """Returns the integer value returned by the function calls to the library in case of IO error. """
    r = call_c_function( physiological_c.status_io_error, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value

def status_decode_error(): 
    """Returns the value returned by the function calls to the library in case of error decoding a file. """
    r = call_c_function( physiological_c.status_decode_error, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value

def status_initialisation_error(): 
    """Returns the value returned by the function calls to the library in case of initialisation error. """
    r = call_c_function( physiological_c.status_initialisation_error, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value

def status_parameter_error(): 
    """Returns the value returned by the function calls to the library in case of parameter error. """
    r = call_c_function( physiological_c.status_parameter_error, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value

def status_unhandled_error(): 
    """Returns the value returned by the function calls to the library in case of unhandled error. """
    r = call_c_function( physiological_c.status_unhandled_error, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value

def test_library(): 
    """Test whether the C library responds. Used in the package testing. """
    number = 134 # a (lucky) number
    descriptor = [  {'name':'input',  'type':'int', 'value':number},
                    {'name':'output', 'type':'int', 'value':None },    ]
    r = call_c_function( physiological_c.echo, descriptor ) 
    return r.output == number






def load_data(filename): 
    """Extract information from Siemens binary physiological data file: load data to global memory structure """ 
    descriptor = [  {'name':'filename',            'type':'string', 'value':filename ,'size':len(filename)}, ]
    r = call_c_function( physiological_c.load_data, descriptor ) 
    if not r.status == status_success(): 
        raise ErrorInCFunction("The execution of load_data was unsuccesful.",r.status,'load_data')
    return r.dictionary 

def get_info(): 
    """Extract information from Siemens binary physiological data file: get information from global memory structure. """ 
    descriptor = [  {'name':'n_samples_breathing', 'type':'uint',     'value':None      }, 
                    {'name':'n_samples_ecg_I',     'type':'uint',     'value':None      }, 
                    {'name':'n_samples_ecg_aVF',   'type':'uint',     'value':None      }, 
                    {'name':'n_samples_pulse_ox',  'type':'uint',     'value':None      }, 
                    {'name':'n_samples_external',  'type':'uint',     'value':None      }, ]
    r = call_c_function( physiological_c.get_info, descriptor ) 
    if not r.status == status_success(): 
        raise ErrorInCFunction("The execution of get_info was unsuccesful.",r.status,'get_info')
    return r.dictionary 

def get_data(n_samples_breathing, n_samples_ecg_I, n_samples_ecg_aVF, n_samples_pulse_ox, n_samples_external): 
    """Extract information from Siemens binary physiological data file: get data from global memory strucutre. """ 
    descriptor = [  {'name':'n_samples_breathing', 'type':'uint',     'value':n_samples_breathing      }, 
                    {'name':'n_samples_ecg_I',     'type':'uint',     'value':n_samples_ecg_I          }, 
                    {'name':'n_samples_ecg_aVF',   'type':'uint',     'value':n_samples_ecg_aVF        }, 
                    {'name':'n_samples_pulse_ox',  'type':'uint',     'value':n_samples_pulse_ox       }, 
                    {'name':'n_samples_external',  'type':'uint',     'value':n_samples_external       }, 
                    {'name':'samples_breathing', 'type':'array',  'value':None, 'dtype':uint16, 'size':(n_samples_breathing)},
                    {'name':'samples_ecg_I',     'type':'array',  'value':None, 'dtype':uint16, 'size':(n_samples_ecg_I)},
                    {'name':'samples_ecg_aVF',   'type':'array',  'value':None, 'dtype':uint16, 'size':(n_samples_ecg_aVF)},
                    {'name':'samples_pulse_ox',  'type':'array',  'value':None, 'dtype':uint16, 'size':(n_samples_pulse_ox)},
                    {'name':'samples_external',  'type':'array',  'value':None, 'dtype':uint16, 'size':(n_samples_external)},
                    {'name':'triggers_breathing','type':'array',  'value':None, 'dtype':uint16, 'size':(n_samples_breathing)},
                    {'name':'triggers_ecg_I',    'type':'array',  'value':None, 'dtype':uint16, 'size':(n_samples_ecg_I)},
                    {'name':'triggers_ecg_aVF',  'type':'array',  'value':None, 'dtype':uint16, 'size':(n_samples_ecg_aVF)},
                    {'name':'triggers_pulse_ox', 'type':'array',  'value':None, 'dtype':uint16, 'size':(n_samples_pulse_ox)},
                    {'name':'triggers_external', 'type':'array',  'value':None, 'dtype':uint16, 'size':(n_samples_external)},                  
                 ]
    r = call_c_function( physiological_c.get_data, descriptor ) 
    if not r.status == status_success(): 
        raise ErrorInCFunction("The execution of get_data was unsuccesful.",r.status,'get_data')
    return r.dictionary 








class Biograph_mMR_Physiology(): 
    def __init__(self, dicom_filename=None):
        self.data = None 
        if dicom_filename is not None: 
            self.load(dicom_filename)

    def load(self,dicom_filename): 
        # FIXME: check if the input is a dicom or a .dat file - now assuming dicom 
        d = dicom.read_file(dicom_filename)
        bindata = d.get(('7fe1', '1010')).value

        fid = open('./physio.dat','wb')  #FIXME: pass the binary buffer directly to the C function 
        fid.write(bindata) 
        fid.close() 

        load_data('./physio.dat')
        info = get_info()
        #print "INFO:",info
        data = get_data(info['n_samples_breathing'],info['n_samples_ecg_I'],info['n_samples_ecg_aVF'],info['n_samples_pulse_ox'],info['n_samples_external'])
        data['sampling_rate_ecg']       = 400;  # FIXME: are these values fixed, or are they stored in the dicom header? 
        data['sampling_rate_breathing'] = 50; 
        data['sampling_rate_external']  = 50; 
        self.data = data
        return self.data  

    def get_data(self):
        return self.data 
