# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 

from __future__ import absolute_import, print_function

from ..Graphs import Dependence, ProbabilisticGraphicalModel
from ..Models import Poisson, Smoothness
from ..Samplers import Sampler
from ..Tracers import RamTracer
from ..Display import DisplayPylab
import numpy

# Define the model components
observation = Poisson('SPECT')
prior_activity = Smoothness('Smoothing_Activity')
prior_attenuation = Smoothness('Smoothing_Attenuation')

# Build the graph 
dag = ProbabilisticGraphicalModel(['activity', 'attenuation', 'counts', 'smoothing-activity', 'smoothing-attenuation'])
dag.set_nodes_given(['counts', 'smoothing-activity', 'smoothing-attenuation'], True)
dag.add_dependence(observation, {'lambda': 'activity', 'alpha': 'attenuation', 'z': 'counts'})
dag.add_dependence(prior_activity, {'x': 'activity', 'beta': 'smoothing-activity'})
dag.add_dependence(prior_attenuation, {'x': 'attenuation', 'beta': 'smoothing-attenuation'})

# Initialize the nodes of the graph   
dag.set_node_value('activity', numpy.ones((10, 10)))

# Instantiate the sampler, tracer and display 
sampler = Sampler(dag)
tracer = RamTracer(sampler)
display = DisplayPylab(tracer)

# Sample 
sampler.sample(1000, trace=False)
display.imagesc_node('activity')

if __name__ == "__main__":
    dag.display_in_browser()
