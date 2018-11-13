import sys 
sys.path.insert(0, '/media/DATA/DOCUMENTI/GITHUB/__OCCIPUT_src/occiput-suite-DEV')
from occiput_suite.simplewrap import *
from occiput_suite.petlink import petlink
from numpy import int32, uint32, uint16, float16, float32
import uuid
from IPython.display import HTML, Javascript, display


__all__ = ['Biograph_mMR_Listmode']


###### IPython integration: ########

LIGHT_BLUE   = 'rgb(200,228,246)'
BLUE         = 'rgb(47,128,246)'
LIGHT_RED    = 'rgb(246,228,200)'
RED          = 'rgb(246,128,47)'
LIGHT_GRAY   = 'rgb(246,246,246)'
GRAY         = 'rgb(200,200,200)'



    
class ProgressBar(): 
    def __init__(self, height='6px', width='100%%', background_color=LIGHT_BLUE, foreground_color=BLUE): 
        self.divid = str(uuid.uuid4())
        self.pb = HTML(
        """
        <div style="border: 1px solid white; width:%s; height:%s; background-color:%s">
            <div id="%s" style="background-color:%s; width:0%%; height:%s"> </div>
        </div> 
        """ % ( width, height, background_color, self.divid, foreground_color, height))
        self.visible = False
    def show(self):
        display(self.pb)
        self.visible = True 
        
    def set_percentage(self,percentage):
        if not self.visible: 
            self.show()
        if percentage < 1:
            percentage = 1
        if percentage > 100:
            percentage = 100 
        display(Javascript("$('div#%s').width('%i%%')" % (self.divid, percentage)))




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



###### Interface C library: ########

# Load library
(found,fullpath,path) = find_c_library("listmode_c",[localpath(), filepath(__file__)]) 
mMR_c = load_c_library(fullpath)


# Utility functions 
def status_success(): 
    """Returns the value returned by the function calls to the library in case of success. """
    r = call_c_function( mMR_c.status_success, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value

def status_io_error(): 
    """Returns the integer value returned by the function calls to the library in case of IO error. """
    r = call_c_function( mMR_c.status_io_error, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value

def status_decode_error(): 
    """Returns the value returned by the function calls to the library in case of error decoding a file. """
    r = call_c_function( mMR_c.status_decode_error, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value

def status_initialisation_error(): 
    """Returns the value returned by the function calls to the library in case of initialisation error. """
    r = call_c_function( mMR_c.status_initialisation_error, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value

def status_parameter_error(): 
    """Returns the value returned by the function calls to the library in case of parameter error. """
    r = call_c_function( mMR_c.status_parameter_error, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value

def status_unhandled_error(): 
    """Returns the value returned by the function calls to the library in case of unhandled error. """
    r = call_c_function( mMR_c.status_unhandled_error, [{'name':'return_value',  'type':'int', 'value':None}] ) 
    return r.return_value


    
# Create interface to the C functions: 
def test_library_mMR_c(): 
    """Test whether the C library mMR_c responds. """
    number = 101 # just a number
    descriptor = [  {'name':'input',  'type':'int', 'value':number},
                    {'name':'output', 'type':'int', 'value':None },    ]
    r = call_c_function( mMR_c.echo, descriptor ) 
    return r.output == number



#
def petlink32_to_dynamic_projection_mMR(filename,n_packets,n_radial_bins,n_angles,n_sinograms,time_bins,n_axial,n_azimuthal,angles_axial,angles_azimuthal,size_u,size_v,n_u,n_v,span,n_segments,segments_sizes,michelogram_segments,michelogram_planes, status_callback): 
    """Make dynamic compressed projection from list-mode data. """
    descriptor = [  {'name':'filename',               'type':'string',  'value':filename ,'size':len(filename)},
                    {'name':'n_packets',              'type':'long',    'value':n_packets                }, 
                    {'name':'n_radial_bins',          'type':'uint',    'value':n_radial_bins            }, 
                    {'name':'n_angles',               'type':'uint',    'value':n_angles                 },
                    {'name':'n_sinograms',            'type':'uint',    'value':n_sinograms              },
                    {'name':'n_time_bins',            'type':'uint',    'value':len(time_bins)-1         },
                    {'name':'time_bins',              'type':'array',   'value':int32(time_bins)         },                                        
                    {'name':'n_axial',                'type':'uint',    'value':n_axial                  },
                    {'name':'n_azimuthal',            'type':'uint',    'value':n_azimuthal              },
                    {'name':'angles_axial',           'type':'array',   'value':angles_axial             },
                    {'name':'angles_azimuthal',       'type':'array',   'value':angles_azimuthal         },
                    {'name':'size_u',                 'type':'float',   'value':size_u                   },
                    {'name':'size_v',                 'type':'float',   'value':size_v                   },
                    {'name':'n_u',                    'type':'uint',    'value':n_u                      },
                    {'name':'n_v',                    'type':'uint',    'value':n_v                      }, 
                    {'name':'span',                   'type':'uint',    'value':span                     }, 
                    {'name':'n_segments',             'type':'uint',    'value':n_segments               }, 
                    {'name':'segments_sizes',         'type':'array',   'value':int32(segments_sizes)    }, 
                    {'name':'michelogram_segments',   'type':'array',   'value':int32(michelogram_segments) }, 
                    {'name':'michelogram_planes',     'type':'array',   'value':int32(michelogram_planes)   }, 
                    {'name':'status_callback',        'type':'function','value':status_callback,  'arg_types':['uint']     },   ]
    r = call_c_function( mMR_c.petlink32_to_dynamic_projection_mMR_michelogram, descriptor ) 
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'petlink32_to_dynamic_projection_mMR_michelogram' was unsuccessful.",r.status,'mMR_c.petlink32_to_dynamic_projection_mMR')
    return r.dictionary 
 

#
def petlink32_to_dynamic_projection_cyclic_mMR(filename, n_packets, n_radial_bins,n_angles,n_sinograms,time_bins,n_axial,n_azimuthal,angles_axial,angles_azimuthal,size_u,size_v,n_u,n_v,span,n_segments,segments_sizes,michelogram_segments,michelogram_planes, status_callback): 
    """Make dynamic compressed projection from list-mode data - cyclic. """
    descriptor = [  {'name':'filename',               'type':'string',  'value':filename ,'size':len(filename)},
                    {'name':'n_packets',              'type':'long',    'value':n_packets                }, 
                    {'name':'n_radial_bins',          'type':'uint',    'value':n_radial_bins            }, 
                    {'name':'n_angles',               'type':'uint',    'value':n_angles                 },
                    {'name':'n_sinograms',            'type':'uint',    'value':n_sinograms              },
                    
                    {'name':'n_frames',               'type':'uint',    'value':time_bins.shape[0]       },
                    {'name':'n_repetitions',          'type':'uint',    'value':time_bins.shape[1]       },
                    {'name':'time_bins',              'type':'array',   'value':int32(time_bins)         }, 
                                                           
                    {'name':'n_axial',                'type':'uint',    'value':n_axial                  },
                    {'name':'n_azimuthal',            'type':'uint',    'value':n_azimuthal              },
                    {'name':'angles_axial',           'type':'array',   'value':angles_axial             },
                    {'name':'angles_azimuthal',       'type':'array',   'value':angles_azimuthal         },
                    {'name':'size_u',                 'type':'float',   'value':size_u                   },
                    {'name':'size_v',                 'type':'float',   'value':size_v                   },
                    {'name':'n_u',                    'type':'uint',    'value':n_u                      },
                    {'name':'n_v',                    'type':'uint',    'value':n_v                      }, 
                    {'name':'span',                   'type':'uint',    'value':span                     }, 
                    {'name':'n_segments',             'type':'uint',    'value':n_segments               }, 
                    {'name':'segments_sizes',         'type':'array',   'value':int32(segments_sizes)    }, 
                    {'name':'michelogram_segments',   'type':'array',   'value':int32(michelogram_segments) }, 
                    {'name':'michelogram_planes',     'type':'array',   'value':int32(michelogram_planes)   }, 
                    {'name':'status_callback',        'type':'function','value':status_callback,  'arg_types':['uint']     },   ]
    r = call_c_function( mMR_c.petlink32_to_dynamic_projection_cyclic_mMR_michelogram, descriptor ) 
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'petlink32_to_dynamic_projection_cyclic_mMR_michelogram' was unsuccessful.",r.status,'mMR_c.petlink32_to_dynamic_projection_cyclic_mMR')
    return r.dictionary 

#
def petline32_load_gates(filename, n_packets, time_bins, status_callback):
    time_bins = int32(time_bins)
    descriptor = [  {'name':'filename',               'type':'string',  'value':filename ,'size':len(filename)},
                    {'name':'n_packets',              'type':'long',    'value':n_packets                }, 
                    {'name':'use_time_bins',          'type':'uint',    'value':len(time_bins)           }, 
                    {'name':'time_bins',              'type':'array',   'value':time_bins                },                                        
                    {'name':'status_callback',        'type':'function','value':status_callback,  'arg_types':['uint']     },   ]
    r = call_c_function( mMR_c.petlink32_load_gates, descriptor ) 
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'petlink32_load_gates' was unsuccessful.",r.status,'mMR_c.petlink32_load_gates')
    return r.dictionary 

#
def get_gates_info(): 
    descriptor = [   {'name':'N_gating1',             'type':'uint',      'value':None       }, 
                     {'name':'N_gating2',             'type':'uint',      'value':None       }, 
                     {'name':'N_gating3',             'type':'uint',      'value':None       }, 
                     {'name':'N_gating4',             'type':'uint',      'value':None       }, 
                     {'name':'N_gating5',             'type':'uint',      'value':None       }, 
                     {'name':'N_gating6',             'type':'uint',      'value':None       }, 
                     {'name':'N_gating7',             'type':'uint',      'value':None       }, 
                     {'name':'N',                     'type':'uint',      'value':None       }, 
                     {'name':'time_start',            'type':'uint',      'value':None       },   
                     {'name':'time_end',              'type':'uint',      'value':None       },   ]                 
    r = call_c_function( mMR_c.get_gates_info, descriptor )
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'get_gates_info' was unsuccessful.",r.status,'mMR_c.get_gates_info')
    return r.dictionary 

#
def get_gates(): 
     info = get_gates_info()
     N = info['N'] 
     descriptor = [  {'name':'N',                     'type':'uint',      'value':N              }, 
                     {'name':'type',                  'type':'array',     'value':None,   'dtype':int32,  'size':(N),       'order':'F'}, 
                     {'name':'time',                  'type':'array',     'value':None,   'dtype':int32,  'size':(N),       'order':'F'}, 
                     {'name':'payload',               'type':'array',     'value':None,   'dtype':int32,  'size':(N),       'order':'F'}  ]
     r = call_c_function( mMR_c.get_gates, descriptor )
     if not r.status == petlink.status_success(): 
         raise ErrorInCFunction("The execution of 'get_gates' was unsuccessful.",r.status,'mMR_c.get_gates')
     return r.dictionary 


#
def get_dynamic_projection_info_prompt(): 
    """Get information about the current dynamic projection data. """
    descriptor = [   {'name':'N_time_bins',             'type':'uint',      'value':None       }, 
                     {'name':'N_counts',                'type':'long',      'value':None       }, 
                     {'name':'N_locations',             'type':'uint',      'value':None       }, 
                     {'name':'compression_ratio',       'type':'float',     'value':None       }, 
                     {'name':'dynamic_inflation',       'type':'float',     'value':None       }, 
                     {'name':'listmode_loss',           'type':'float',     'value':None       }, 
                     {'name':'time_start',              'type':'uint',      'value':None       },   
                     {'name':'time_end',                'type':'uint',      'value':None       },   ]                 
    r = call_c_function( mMR_c.get_dynamic_projection_info_prompt, descriptor )
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'get_dynamic_projection_info_prompt' was unsuccessful.",r.status,'mMR_c.get_dynamic_projection_info_prompt')
    return r.dictionary 


#
def get_dynamic_projection_info_delay(): 
    """Get information about the current dynamic projection data. """
    descriptor = [   {'name':'N_time_bins',             'type':'uint',      'value':None       }, 
                     {'name':'N_counts',                'type':'long',      'value':None       }, 
                     {'name':'N_locations',             'type':'uint',      'value':None       }, 
                     {'name':'compression_ratio',       'type':'float',     'value':None       }, 
                     {'name':'dynamic_inflation',       'type':'float',     'value':None       }, 
                     {'name':'listmode_loss',           'type':'float',     'value':None       }, 
                     {'name':'time_start',              'type':'uint',      'value':None       },   
                     {'name':'time_end',                'type':'uint',      'value':None       },   ]                 
    r = call_c_function( mMR_c.get_dynamic_projection_info_delay, descriptor )
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'get_dynamic_projection_info_delay' was unsuccessful.",r.status,'mMR_c.get_dynamic_projection_info_delay')
    return r.dictionary 


#
def get_static_projection_info_prompt(time_bin): 
    """Get information about a time bin of the current dynamic projection data. """
    descriptor = [  {'name':'time_bin',                'type':'uint',      'value':time_bin       }, 
                    {'name':'time_start',              'type':'uint',      'value':None           }, 
                    {'name':'time_end',                'type':'uint',      'value':None           }, 
                    {'name':'N_counts',                'type':'uint',      'value':None           }, 
                    {'name':'N_locations',             'type':'uint',      'value':None           }, 
                    {'name':'compression_ratio',       'type':'float',     'value':None           },
                    {'name':'listmode_loss',           'type':'float',     'value':None           },
                    {'name':'N_axial',                 'type':'uint',      'value':None           }, 
                    {'name':'N_azimuthal',             'type':'uint',      'value':None           }, 
                    {'name':'angles_axial',            'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000) }, 
                    {'name':'angles_azimuthal',        'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000) }, 
                    {'name':'size_u',                  'type':'float',     'value':None           }, 
                    {'name':'size_v',                  'type':'float',     'value':None           }, 
                    {'name':'N_u',                     'type':'uint',      'value':None           }, 
                    {'name':'N_v',                     'type':'uint',      'value':None           },  ] 
    r = call_c_function( mMR_c.get_static_projection_info_prompt, descriptor )
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'get_static_projection_info_prompt' was unsuccessful.",r.status,'mMR_c.get_static_projection_info_prompt')
    return r.dictionary 


#
def get_static_projection_info_delay(time_bin): 
    """Get information about a time bin of the current dynamic projection data. """
    descriptor = [  {'name':'time_bin',                'type':'uint',      'value':time_bin       }, 
                    {'name':'time_start',              'type':'uint',      'value':None           }, 
                    {'name':'time_end',                'type':'uint',      'value':None           }, 
                    {'name':'N_counts',                'type':'uint',      'value':None           }, 
                    {'name':'N_locations',             'type':'uint',      'value':None           }, 
                    {'name':'compression_ratio',       'type':'float',     'value':None           },
                    {'name':'listmode_loss',           'type':'float',     'value':None           },
                    {'name':'N_axial',                 'type':'uint',      'value':None           }, 
                    {'name':'N_azimuthal',             'type':'uint',      'value':None           }, 
                    {'name':'angles_axial',            'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000) }, 
                    {'name':'angles_azimuthal',        'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000) }, 
                    {'name':'size_u',                  'type':'float',     'value':None           }, 
                    {'name':'size_v',                  'type':'float',     'value':None           }, 
                    {'name':'N_u',                     'type':'uint',      'value':None           }, 
                    {'name':'N_v',                     'type':'uint',      'value':None           },  ] 
    r = call_c_function( mMR_c.get_static_projection_info_delay, descriptor )
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'get_static_projection_info_delay' was unsuccessful.",r.status,'mMR_c.get_static_projection_info_delay')
    return r.dictionary 
    

#
def get_global_static_projection_info_prompt(): 
    """Get information about a time bin of the current dynamic projection data. """
    descriptor = [  {'name':'time_start',              'type':'uint',      'value':None           }, 
                    {'name':'time_end',                'type':'uint',      'value':None           }, 
                    {'name':'N_counts',                'type':'uint',      'value':None           }, 
                    {'name':'N_locations',             'type':'uint',      'value':None           }, 
                    {'name':'compression_ratio',       'type':'float',     'value':None           },
                    {'name':'listmode_loss',           'type':'float',     'value':None           },
                    {'name':'N_axial',                 'type':'uint',      'value':None           }, 
                    {'name':'N_azimuthal',             'type':'uint',      'value':None           }, 
                    {'name':'angles_axial',            'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000), 'order':'F' }, 
                    {'name':'angles_azimuthal',        'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000), 'order':'F' },  
                    {'name':'size_u',                  'type':'float',     'value':None           }, 
                    {'name':'size_v',                  'type':'float',     'value':None           }, 
                    {'name':'N_u',                     'type':'uint',      'value':None           }, 
                    {'name':'N_v',                     'type':'uint',      'value':None           },  ] 
    r = call_c_function( mMR_c.get_global_static_projection_info_prompt, descriptor )
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'get_global_static_projection_info_prompt' was unsuccessful.",r.status,'mMR_c.get_global_static_projection_info_prompt')
    return r.dictionary 


#
def get_global_static_projection_info_delay(): 
    """Get information about a time bin of the current dynamic projection data. """
    descriptor = [  {'name':'time_start',              'type':'uint',      'value':None           }, 
                    {'name':'time_end',                'type':'uint',      'value':None           }, 
                    {'name':'N_counts',                'type':'uint',      'value':None           }, 
                    {'name':'N_locations',             'type':'uint',      'value':None           }, 
                    {'name':'compression_ratio',       'type':'float',     'value':None           },
                    {'name':'listmode_loss',           'type':'float',     'value':None           },
                    {'name':'N_axial',                 'type':'uint',      'value':None           }, 
                    {'name':'N_azimuthal',             'type':'uint',      'value':None           }, 
                    {'name':'angles_axial',            'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000), 'order':'F' }, 
                    {'name':'angles_azimuthal',        'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000), 'order':'F' },  
                    {'name':'size_u',                  'type':'float',     'value':None           }, 
                    {'name':'size_v',                  'type':'float',     'value':None           }, 
                    {'name':'N_u',                     'type':'uint',      'value':None           }, 
                    {'name':'N_v',                     'type':'uint',      'value':None           },  ] 
    r = call_c_function( mMR_c.get_global_static_projection_info_delay, descriptor )
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'get_global_static_projection_info_delay' was unsuccessful.",r.status,'mMR_c.get_global_static_projection_info_delay')
    return r.dictionary 
 

#
def get_static_projection_prompt(time_bin): 
     """Get the static projection corresponding to a given time bin of the current dynamic projection data. """
     info = get_static_projection_info_prompt(time_bin) 
     N_locations = info['N_locations']
     N_axial     = info['N_axial']
     N_azimuthal = info['N_azimuthal']
     descriptor = [  {'name':'time_bin',                'type':'uint',      'value':time_bin       }, 
                     {'name':'time_start',              'type':'uint',      'value':None           }, 
                     {'name':'time_end',                'type':'uint',      'value':None           }, 
                     {'name':'N_counts',                'type':'uint',      'value':None           }, 
                     {'name':'N_locations',             'type':'uint',      'value':None           }, 
                     {'name':'compression_ratio',       'type':'float',     'value':None           },
                     {'name':'listmode_loss',           'type':'float',     'value':None           },
                     {'name':'N_axial',                 'type':'uint',      'value':None           }, 
                     {'name':'N_azimuthal',             'type':'uint',      'value':None           }, 
                     {'name':'angles_axial',            'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000) }, 
                     {'name':'angles_azimuthal',        'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000) },  
                     {'name':'size_u',                  'type':'float',     'value':None           }, 
                     {'name':'size_v',                  'type':'float',     'value':None           }, 
                     {'name':'N_u',                     'type':'uint',      'value':None           }, 
                     {'name':'N_v',                     'type':'uint',      'value':None           },  
                     {'name':'offsets',                 'type':'array',     'value':None,   'dtype':int32,  'size':(N_azimuthal,N_axial), 'order':'F'}, 
                     {'name':'counts',                  'type':'array',     'value':None,   'dtype':float32,  'size':(N_locations),       'order':'F'}, 
                     {'name':'locations',               'type':'array',     'value':None,   'dtype':uint16,  'size':(3,N_locations),      'order':'F'},             ] 
     r = call_c_function( mMR_c.get_static_projection_prompt, descriptor )
     if not r.status == petlink.status_success(): 
         raise ErrorInCFunction("The execution of 'get_static_projection_prompt' was unsuccessful.",r.status,'mMR_c.get_static_projection_prompt')
     return r.dictionary 


#
def get_static_projection_delay(time_bin): 
     """Get the static projection corresponding to a given time bin of the current dynamic projection data. """
     info = get_static_projection_info_delay(time_bin) 
     N_locations = info['N_locations']
     N_axial     = info['N_axial']
     N_azimuthal = info['N_azimuthal']
     descriptor = [  {'name':'time_bin',                'type':'uint',      'value':time_bin       }, 
                     {'name':'time_start',              'type':'uint',      'value':None           }, 
                     {'name':'time_end',                'type':'uint',      'value':None           }, 
                     {'name':'N_counts',                'type':'uint',      'value':None           }, 
                     {'name':'N_locations',             'type':'uint',      'value':None           }, 
                     {'name':'compression_ratio',       'type':'float',     'value':None           },
                     {'name':'listmode_loss',           'type':'float',     'value':None           },
                     {'name':'N_axial',                 'type':'uint',      'value':None           }, 
                     {'name':'N_azimuthal',             'type':'uint',      'value':None           }, 
                     {'name':'angles_axial',            'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000) }, 
                     {'name':'angles_azimuthal',        'type':'float',     'value':None,    'dtype':float32,   'size':(1,10000) },  
                     {'name':'size_u',                  'type':'float',     'value':None           }, 
                     {'name':'size_v',                  'type':'float',     'value':None           }, 
                     {'name':'N_u',                     'type':'uint',      'value':None           }, 
                     {'name':'N_v',                     'type':'uint',      'value':None           },  
                     {'name':'offsets',                 'type':'array',     'value':None,   'dtype':int32,  'size':(N_azimuthal,N_axial), 'order':'F'}, 
                     {'name':'counts',                  'type':'array',     'value':None,   'dtype':float32,  'size':(N_locations),       'order':'F'}, 
                     {'name':'locations',               'type':'array',     'value':None,   'dtype':uint16,  'size':(3,N_locations),      'order':'F'},             ] 
     r = call_c_function( mMR_c.get_static_projection_delay, descriptor )
     if not r.status == petlink.status_success(): 
         raise ErrorInCFunction("The execution of 'get_static_projection_delay' was unsuccessful.",r.status,'mMR_c.get_static_projection_delay')
     return r.dictionary 


#
def get_global_static_projection_prompt(): 
    """Get the global static projection (i.e. no time binning). """
    info = get_global_static_projection_info_prompt() 
    N_locations = info['N_locations']
    N_axial     = info['N_axial']
    N_azimuthal = info['N_azimuthal']
    descriptor = [  {'name':'time_start',              'type':'uint',      'value':None           }, 
                    {'name':'time_end',                'type':'uint',      'value':None           }, 
                    {'name':'N_counts',                'type':'uint',      'value':None           }, 
                    {'name':'N_locations',             'type':'uint',      'value':None           }, 
                    {'name':'compression_ratio',       'type':'float',     'value':None           },
                    {'name':'listmode_loss',           'type':'float',     'value':None           },
                    {'name':'N_axial',                 'type':'uint',      'value':None           }, 
                    {'name':'N_azimuthal',             'type':'uint',      'value':None           }, 
                    {'name':'angles_axial',            'type':'float',     'value':None,    'dtype':float32,   'size':(1,N_axial) }, 
                    {'name':'angles_azimuthal',        'type':'float',     'value':None,    'dtype':float32,   'size':(1,N_azimuthal) }, 
                    {'name':'size_u',                  'type':'float',     'value':None           }, 
                    {'name':'size_v',                  'type':'float',     'value':None           }, 
                    {'name':'N_u',                     'type':'uint',      'value':None           }, 
                    {'name':'N_v',                     'type':'uint',      'value':None           },  
                    {'name':'offsets',                 'type':'array',     'value':None,   'dtype':int32,  'size':(N_azimuthal,N_axial), 'order':'F'}, 
                    {'name':'counts',                  'type':'array',     'value':None,   'dtype':float32, 'size':(N_locations),        'order':'F'}, 
                    {'name':'locations',               'type':'array',     'value':None,   'dtype':uint16,  'size':(3,N_locations),      'order':'F'},             ] 
    r = call_c_function( mMR_c.get_global_static_projection_prompt, descriptor )
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'get_global_static_projection_prompt' was unsuccessful.",r.status,'mMR_c.get_global_static_projection_prompt')
    return r.dictionary     


#
def get_global_static_projection_delay(): 
    """Get the global static projection (i.e. no time binning). """
    info = get_global_static_projection_info_delay() 
    N_locations = info['N_locations']
    N_axial     = info['N_axial']
    N_azimuthal = info['N_azimuthal']
    descriptor = [  {'name':'time_start',              'type':'uint',      'value':None           }, 
                    {'name':'time_end',                'type':'uint',      'value':None           }, 
                    {'name':'N_counts',                'type':'uint',      'value':None           }, 
                    {'name':'N_locations',             'type':'uint',      'value':None           }, 
                    {'name':'compression_ratio',       'type':'float',     'value':None           },
                    {'name':'listmode_loss',           'type':'float',     'value':None           },
                    {'name':'N_axial',                 'type':'uint',      'value':None           }, 
                    {'name':'N_azimuthal',             'type':'uint',      'value':None           }, 
                    {'name':'angles_axial',            'type':'float',     'value':None,    'dtype':float32,   'size':(1,N_axial) }, 
                    {'name':'angles_azimuthal',        'type':'float',     'value':None,    'dtype':float32,   'size':(1,N_azimuthal) }, 
                    {'name':'size_u',                  'type':'float',     'value':None           }, 
                    {'name':'size_v',                  'type':'float',     'value':None           }, 
                    {'name':'N_u',                     'type':'uint',      'value':None           }, 
                    {'name':'N_v',                     'type':'uint',      'value':None           },  
                    {'name':'offsets',                 'type':'array',     'value':None,   'dtype':int32,  'size':(N_azimuthal,N_axial), 'order':'F'}, 
                    {'name':'counts',                  'type':'array',     'value':None,   'dtype':float32, 'size':(N_locations),        'order':'F'}, 
                    {'name':'locations',               'type':'array',     'value':None,   'dtype':uint16,  'size':(3,N_locations),      'order':'F'},             ] 
    r = call_c_function( mMR_c.get_global_static_projection_delay, descriptor )
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'get_global_static_projection_delay' was unsuccessful.",r.status,'mMR_c.get_global_static_projection_delay')
    return r.dictionary 


#
def free_memory(): 
    """Free memory""" 
    r = call_c_function( mMR_c.free_memory, [] )
    if not r.status == petlink.status_success(): 
        raise ErrorInCFunction("The execution of 'free_memory' was unsuccessful.",r.status,'mMR_c.free_memory')
    return r.dictionary 

    



    


class Biograph_mMR_Listmode(): 
    def __init__(self):
        self.name         = "Siemens Biograph mMR" 
        self.manufacturer = "Siemens" 
        self.version      = "0.0" 

    def load_listmode(self, listmode_data_filename, n_packets, time_bins, binning, n_radial_bins, n_angles, n_sinograms, span, segments_sizes, michelogram_segments, michelogram_planes, status_callback=None): 
        n_axial                = binning.N_axial 
        n_azimuthal            = binning.N_azimuthal 
        angles_axial           = binning.angles_axial 
        angles_azimuthal       = binning.angles_azimuthal 
        size_u                 = binning.size_u 
        size_v                 = binning.size_v 
        n_u                    = binning.N_u 
        n_v                    = binning.N_v 
        n_segments = len(segments_sizes)*2-1
        if status_callback is None: 
            status_callback = lambda x: None
        petlink32_to_dynamic_projection_mMR(listmode_data_filename,n_packets, n_radial_bins, n_angles,n_sinograms,time_bins,n_axial,n_azimuthal,angles_axial,angles_azimuthal,size_u,size_v,n_u,n_v,span,n_segments,segments_sizes,michelogram_segments, michelogram_planes,status_callback)
        info_prompt = get_dynamic_projection_info_prompt()  
        info_delay = get_dynamic_projection_info_delay()        #FIXME:delay - also return information related to delays 
        return info_prompt

    def load_gates(self, listmode_data_filename, time_range_ms, n_packets=100000000000, status_callback=None): 
        if status_callback is None: 
            status_callback = lambda x: None
        petline32_load_gates(listmode_data_filename, n_packets, time_range_ms, status_callback) 
        info = get_gates_info()
        return info 

    def get_gates(self):
        return get_gates() 

    def load_listmode_cyclic(self, listmode_data_filename, time_bins, binning, n_radial_bins, n_angles, n_sinograms, span, segments_sizes, michelogram_segments, michelogram_planes, n_packets=100000000000, status_callback=None): 
        n_axial                = binning.N_axial 
        n_azimuthal            = binning.N_azimuthal 
        angles_axial           = binning.angles_axial 
        angles_azimuthal       = binning.angles_azimuthal 
        size_u                 = binning.size_u 
        size_v                 = binning.size_v 
        n_u                    = binning.N_u 
        n_v                    = binning.N_v 
        n_segments = len(segments_sizes)*2-1
        if status_callback is None: 
            status_callback = lambda x: None
        petlink32_to_dynamic_projection_cyclic_mMR(listmode_data_filename, n_packets, n_radial_bins, n_angles, n_sinograms, time_bins, n_axial, n_azimuthal, angles_axial, angles_azimuthal, size_u,size_v,n_u,n_v,span,n_segments,segments_sizes,michelogram_segments, michelogram_planes,status_callback)
        info_prompt = get_dynamic_projection_info_prompt()  
        info_delay = get_dynamic_projection_info_delay()        #FIXME:delay - also return information related to delays 
        return info_prompt

    def get_measurement_prompt(self,time_bin):             
        return get_static_projection_prompt(time_bin) 
    
    def get_measurement_prompt_info(self,time_bin):
        return get_static_projection_info_prompt(time_bin)

    def get_measurement_static_prompt(self):  
        return get_global_static_projection_prompt()   
    
    def get_measurement_static_prompt_info(self): 
        return get_global_static_projection_info_prompt()

    def get_measurement_delay(self,time_bin):             
        return get_static_projection_delay(time_bin) 

    def get_measurement_delay_info(self,time_bin):
        return get_static_projection_info_delay(time_bin)

    def get_measurement_static_delay(self): 
        return get_global_static_projection_delay()  

    def get_measurement_static_delay_info(self): 
        return get_global_static_projection_info_delay()

    #def __del__(self):        
    def free_memory(self): 
        free_memory() 
        print "mMR Listmode data structures deallocated. "  


