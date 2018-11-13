# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 

from __future__ import absolute_import, print_function

from ..Graphs import Dependence, ProbabilisticGraphicalModel
from ..Models import MultivariateGaussian
from ..Samplers import Sampler
from ..Tracers import RamTracer
from ..Display import DisplayPylab
import numpy

# Define the model
ndim = 10
model = MultivariateGaussian('gaussian')

# Build the graph 
dag = ProbabilisticGraphicalModel(['x', 'mu', 'cov'])
dag.set_nodes_given(['mu', 'cov'], True)
dag.add_dependence(model, {'x': 'x', 'mu': 'mu', 'cov': 'cov'})

# Initialize the nodes of the graph
dag.set_node_value('x', numpy.ones((1, ndim)))
dag.set_node_value('mu', numpy.zeros((1, ndim)))
dag.set_node_value('cov', numpy.eye(ndim))

# Initialise sampler, tracer, display 
sampler = Sampler(dag)
tracer = RamTracer(sampler)
display = DisplayPylab(tracer)

# Sample 
sampler.sample(1000, trace=False)
# display.plot('mu')
# display.plot('sigma')


if __name__ == "__main__":
    dag.display_in_browser()
