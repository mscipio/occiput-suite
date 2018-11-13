# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 
# Harvard University, Martinos Center for Biomedical Imaging 
# Dec 2013, Boston 

from __future__ import absolute_import, print_function

__all__ = ['Sampler']

from .Graphs import Node, ProbabilisticGraphicalModel, name
from .Tracers import RamTracer
from .exceptions import *
from .verbose import *
from .accessories import ProgressBar, LIGHT_RED, RED

import numpy
import inspect, sys
import ipy_table
import pickle

try:
    import scipy
except:
    print("Please install Scipy")
    has_scipy = False
else:
    has_scipy = True
    from scipy.optimize import fmin_l_bfgs_b

MAX_RANDOM_NODES = 10000


# Sampling strategies
class SamplingStrategy(object):
    def __init__(self, graph):
        self.graph = graph

    def next_node(self):
        pass


class RandomNodesStrategy(SamplingStrategy):
    def __init__(self, *args, **kwds):
        super(RandomNodesStrategy, self).__init__(*args, **kwds)

    def next_node(self):
        nodes = self.graph.get_nodes()
        node = None
        for i in range(MAX_RANDOM_NODES):
            index = numpy.random.randint(len(nodes))
            if not nodes[index].is_given():
                node = nodes[index]
                break
        if node is None:
            print_important(
                "RandomNodesStrategy could not find a node after %d attempts, are you sure that the model has unobserved variables?" % (
                MAX_RANDOM_NODES))
        return nodes[index]


# Sampling methods
class RandomNumberGenerator():
    """Random number generator class. It includes all the functions used by the samplers for generation of random numbers. """

    def __init__(self):
        pass

    def rand_normal(self, mu, std):
        return numpy.random.normal(mu, std)

    def rand_multivariate_normal(self, mu, cov):
        return numpy.random.multivariate_normal(mu, cov)

    def rand_uniform(self):
        return numpy.random.rand()


class SamplingMethod(object):
    def __init__(self):
        self._requires_log_probability = False
        self._requires_log_probability_gradient = False
        self._requires_log_probability_hessian = False
        self._requires_log_probability_diagonal_hessian = False
        self._requires_own_sampler = False
        self._is_optimizer = True
        # multi-purpose random number generator: 
        self.random_number_generator = RandomNumberGenerator()

    def requires_log_probability(self):
        return self._requires_log_probability

    def requires_log_probability_gradient(self):
        return self._requires_log_probability_gradient

    def requires_log_probability_hessian(self):
        return self._requires_log_probability_hessian

    def requires_log_probability_diagonal_hessian(self):
        return self._requires_log_probability_diagonal_hessian

    def requires_own_sampler(self):
        return self._requires_own_sampler

    def is_optimizer(self):
        return self._is_optimizer

    def can_be_used_with_node(self, node):
        if not isinstance(node, Node):
            raise UnexpectedParameterType("node must be an instance of Node.")
        if self.requires_own_sampler():
            if not node.can_sample_conditional_probability():
                return False
        if self.requires_log_probability():
            if not node.has_log_conditional_probability():
                return False
        if self.requires_log_probability_gradient():
            if not node.has_log_conditional_probability_gradient():
                return False
        if self.requires_log_probability_hessian():
            if not node.has_log_conditional_probability_hessian():
                return False
        if self.requires_log_probability_diagonal_hessian():
            if not node.has_log_conditional_probability_diagonal_hessian():
                return False
        return True

    def get_name(self):
        if hasattr(self, "__name__"):
            return self.__name__
        elif hasattr(self, "__class__"):
            return self.__class__.__name__
        else:
            raise Exception("This should never ever happen. ")

            # Subclass this function:

    def sample(self, graph, node, parameters=None):
        # FIXME: implement
        print_debug("sample() not defined. ")
        return None

    # Subclass this function:
    def default_parameters(self):
        return {}


class AncestralSampling(SamplingMethod):
    def __init__(self, *args, **kwds):
        super(AncestralSampling, self).__init__(*args, **kwds)
        self._requires_own_sampler = True
        self._is_optimizer = False

    def sample(self, graph, node, parameters=None):
        print_debug("Sampling node '%s' with %s .." % (name(node), name(self)))
        sample = graph.sample_conditional_probability_node(node)
        return sample

    def default_parameters(self):
        return {}


class GradientAscent(SamplingMethod):
    def __init__(self, *args, **kwds):
        super(GradientAscent, self).__init__(*args, **kwds)
        self._requires_log_probability = True
        self._requires_log_probability_gradient = True
        self._is_optimizer = True

    def sample(self, graph, node, parameters=None):
        print_debug("Sampling node '%s' with %s ..\n" % (name(node), name(self)))
        if parameters is None:
            parameters = self.default_parameters()
        sample = graph.get_node_value(node)
        gradient = graph.get_log_conditional_probability_gradient(node, sample)
        sample = sample + parameters['step_size'] * gradient
        return sample

    def default_parameters(self):
        return {'step_size': 0.1}


class ExpectationMaximization(SamplingMethod):
    def __init__(self, *args, **kwds):
        super(ExpectationMaximization, self).__init__(*args, **kwds)
        self._requires_log_probability = True
        self._requires_log_probability_gradient = True
        self._is_optimizer = True

    def sample(self, graph, node, parameters=None):
        print_debug("Sampling node '%s' with %s ..\n" % (name(node), name(self)))
        if parameters is None:
            parameters = self.default_parameters()
        sample = graph.get_node_value(node)
        gradient = graph.get_log_conditional_probability_gradient(node, sample)
        sample = sample + gradient
        return sample

    def default_parameters(self):
        return {}


class QuasiNewton_L_BFGS_B(SamplingMethod):
    def __init__(self, *args, **kwds):
        super(QuasiNewton_L_BFGS_B, self).__init__(*args, **kwds)
        self._requires_log_probability = True
        self._requires_log_probability_gradient = True
        self._is_optimizer = True

    def sample(self, graph, node, parameters=None):
        print_debug("Sampling node '%s' with %s ..\n" % (name(node), name(self)))
        if parameters is None:
            parameters = self.default_parameters()

        approximation_terms = parameters['approximation_terms']
        factr = parameters['factr']
        pgtol = parameters['pgtol']
        maxfun = parameters['maxfun']
        self._graph = graph
        self._node = node
        bounds = parameters['bounds']
        sample = numpy.float64(graph.get_node_value(node))

        if bounds is None:
            bounds = [(None, None)] * sample.size
        elif bounds == 'nonnegative':
            bounds = [(0, None)] * sample.size
        else:
            raise ParameterError("'bounds' parameter for QuasiNewton_L_BFGS_B.")

        sample, f, d = fmin_l_bfgs_b(self._func, sample, fprime=self._grad, args=(), bounds=bounds,
                                     m=approximation_terms, factr=factr, pgtol=pgtol, maxfun=maxfun)
        print_debug("L_BFGS_B: %s (%d function calls)." % (d['task'], d['funcalls']))

        sample = numpy.asarray(sample)
        return sample

    def _func(self, sample, *args):
        sample = numpy.asarray(sample)
        self._graph.set_node_value(self._node,
                                   sample)  # FIXME: this line must be removed, but then there is a problem with l_bfgs_b
        return -1. * numpy.float64(self._graph.get_log_conditional_probability(self._node))

    def _grad(self, sample, *args):
        sample = numpy.asarray(sample)
        self._graph.set_node_value(self._node,
                                   sample)  # FIXME: this line must be removed, but then there is a problem with l_bfgs_b
        return -1. * numpy.float64(self._graph.get_log_conditional_probability_gradient(self._node))

    def default_parameters(self):
        return {'approximation_terms': 10, 'factr': 1e6, 'pgtol': 1e-8, 'maxfun': 15000, 'bounds': None}


class Newton(SamplingMethod):
    """Newton method, optimisation"""

    def __init__(self, *args, **kwds):
        super(Newton, self).__init__(*args, **kwds)
        self._requires_log_probability = True
        self._requires_log_probability_gradient = True
        self._requires_log_probability_hessian = True
        self._is_optimizer = True

    def sample(self, graph, node, parameters=None):
        print_debug("Sampling node '%s' with %s ..\n" % (name(node), name(self)))
        if parameters is None:
            parameters = self.default_parameters()
        sample = graph.get_node_value(node)
        gradient = graph.get_log_conditional_probability_gradient(node, sample)
        hessian = graph.get_log_conditional_probability_hessian(node, sample)
        sample = sample + parameters['step_size'] * numpy.dot(numpy.linalg.inv(hessian), gradient.T).T
        return sample

    def default_parameters(self):
        return {'step_size': 0.99}


class MetropolisHastingsMCMC(SamplingMethod):
    def __init__(self, *args, **kwds):
        super(MetropolisHastingsMCMC, self).__init__(*args, **kwds)
        self._requires_log_probability = True
        self._is_optimizer = False

    def sample(self, graph, node, parameters=None):
        print_debug("Sampling node '%s' with %s ..\n" % (name(node), name(self)))
        if parameters is None:
            parameters = self.default_parameters()
        sigma = parameters['sigma']
        sample = graph.get_node_value(node)

        candidate = sample + self.random_number_generator.rand_normal(0, sigma * numpy.ones(sample.shape))
        alpha = min(1, numpy.exp(
            graph.get_log_conditional_probability(node, candidate) - graph.get_log_conditional_probability(node,
                                                                                                           sample)))
        u = self.random_number_generator.rand_uniform()
        if u < alpha:
            accept = True
            sample = candidate
        else:
            accept = False
            # sample = last_sample
        return sample

    def default_parameters(self):
        return {'sigma': 1}


class HamiltonianMCMC(SamplingMethod):
    def __init__(self, *args, **kwds):
        super(HamiltonianMCMC, self).__init__(*args, **kwds)
        self._requires_log_probability = True
        self._is_optimizer = False

    def sample(self, graph, node, parameters=None):
        # todo: implement!!
        print_debug("Sampling node '%s' with %s ..\n" % (name(node), name(self)))
        if parameters is None:
            parameters = self.default_parameters()
        sigma = parameters['sigma']
        sample = graph.get_node_value(node)

        candidate = sample + self.random_number_generator.rand_normal(0, sigma * numpy.ones(sample.shape))
        alpha = min(1, numpy.exp(
            graph.get_log_conditional_probability(node, candidate) - graph.get_log_conditional_probability(node,
                                                                                                           sample)))
        u = self.random_number_generator.rand_uniform()
        if u < alpha:
            accept = True
            sample = candidate
        else:
            accept = False
            # sample = last_sample
        return sample

    def default_parameters(self):
        return {'sigma': 1}


class LangevinAdjustedMetropolisHastingsMCMC(SamplingMethod):
    def __init__(self, *args, **kwds):
        super(LangevinAdjustedMetropolisHastingsMCMC, self).__init__(*args, **kwds)
        self._requires_log_probability = True
        # self._requires_log_probability_gradient = True
        # self._requires_log_probability_hessian = True
        self._is_optimizer = False

    def sample(self, graph, node, parameters=None):
        # todo: implement!!
        print_debug("Sampling node '%s' with %s ..\n" % (name(node), name(self)))
        if parameters is None:
            parameters = self.default_parameters()
        sigma = parameters['sigma']
        sample = graph.get_node_value(node)

        candidate = sample + self.random_number_generator.rand_normal(0, sigma * numpy.ones(sample.shape))
        alpha = min(1, numpy.exp(
            graph.get_log_conditional_probability(node, candidate) - graph.get_log_conditional_probability(node,
                                                                                                           sample)))
        u = self.random_number_generator.rand_uniform()
        if u < alpha:
            accept = True
            sample = candidate
        else:
            accept = False
            # sample = last_sample
        return sample

    def default_parameters(self):
        return {'sigma': 1}


def best_sampling_method(graph, node, sampling_methods):
    """Select the best sampling method out of a list of sampling methods. """
    # Use the preference expressed by the node and the properties of the node (gradient, ..) and of the sampling methods. 
    # 1) if available, use the node-specific preference:
    # FIXME: implement node preference 
    # 2) choose according to absolute preference criterium:   
    #   1- prefer a method based on direct sampling if node has a direct sampling method
    sampling_method = None
    if node.can_sample_conditional_probability():
        for method in sampling_methods:
            if method.requires_own_sampler():
                sampling_method == method
                #   2- prefer posterior sampling to optimisation:
    # 3- prefer second order sampling strategies to first order (gradient based)
    #   4- prefer first order to order zero (likelihood based): 
    optimisers = []
    samplers = []
    if sampling_method is None:
        for method in sampling_methods:
            if method.is_optimizer():
                optimisers.append(method)
            else:
                samplers.append(method)
        if node.has_log_conditional_probability_hessian():
            for method in samplers:
                if method.requires_log_probability_hessian():
                    sampling_method = method
        if sampling_method is None:
            if node.has_log_conditional_probability_diagonal_hessian():
                for method in samplers:
                    if method.requires_log_probability_diagonal_hessian():
                        sampling_method = method
            if sampling_method is None:
                if node.has_log_conditional_probability_gradient():
                    for method in samplers:
                        if method.requires_log_probability_gradient():
                            sampling_method = method
                if sampling_method is None:
                    if node.has_log_conditional_probability():
                        for method in samplers:
                            if method.requires_log_probability():
                                sampling_method = method
                    if sampling_method is None:
                        if node.has_log_conditional_probability_hessian():
                            for method in optimisers:
                                if method.requires_log_probability_hessian():
                                    sampling_method = method
                        if sampling_method is None:
                            if node.has_log_conditional_probability_diagonal_hessian():
                                for method in optimisers:
                                    if method.requires_log_probability_diagonal_hessian():
                                        sampling_method = method
                            if sampling_method is None:
                                if node.has_log_conditional_probability_gradient():
                                    for method in optimisers:
                                        if method.requires_log_probability_gradient():
                                            sampling_method = method
                                if sampling_method is None:
                                    if node.has_log_conditional_probability():
                                        for method in optimisers:
                                            if method.requires_log_probability():
                                                sampling_method = method
    print_debug("Best method for node %s: %s" % (name(node), name(sampling_method)))
    return sampling_method


# Sampler state
class SamplerStateNode():
    """SamplerStateNode: stores the information relative to the sampling of a node. """

    def __init__(self, node_name):
        self.node_name = node_name
        self.reset_state()

    def reset_state(self):
        self.n_accepted = 0
        self.n_rejected = 0
        self.n_samples = 0
        self.rejection_ratio = 0.0
        self.sample_ratio = 0.0  # ratio between number of samples of the node and number of samples of the graph
        self.last_method = None
        self.last_threshold = None
        self.last_random_number = None
        self.last_seed = None
        self.last_parameters = None
        self.last_accepted = None

    def to_dictionary(self):
        dictionary = {
            'node_name': self.node_name,
            'n_accepted': self.n_accepted,
            'n_rejected': self.n_rejected,
            'n_samples': self.n_samples,
            'rejection_ratio': self.rejection_ratio,
            'sample_ratio': self.sample_ratio,
            'last_threshold': self.last_threshold,
            'last_random_number': self.last_random_number,
            'last_seed': self.last_seed,
            'last_parameters': self.last_parameters,
            'last_accepted': self.last_accepted,
            'last_method': self.last_method}
        return dictionary

    def to_pickle(self):
        dictionary = self.to_dictionary()
        return pickle.dumps(dictionary)

    def new_sample(self, method, accepted, threshold, random_number, seed, parameters, nsamples_graph):
        self.n_samples += 1
        if accepted:
            self.n_accepted += 1
        else:
            self.n_rejected += 1
        self.last_method = method
        self.last_threshold = threshold
        self.last_random_number = random_number
        self.last_seed = seed
        self.last_parameters = parameters
        self.last_accepted = bool(accepted)
        self.rejection_ratio = self.n_rejected / (1.0 * self.n_samples)
        self.sample_ratio = nsamples_graph / (1.0 * self.n_samples)

    def _repr_html_(self):
        table_data = [['node_name', self.node_name, "name of the node"],
                      ['last_method', self.last_method, "sampling method used to obtain the last sample"],
                      ['n_samples', self.n_samples, "total number of samples"],
                      ['n_accepted', self.n_accepted, "number of samples accepted"],
                      ['n_rejected', self.n_rejected, "number of samples rejected"],
                      ['rejection_ratio', self.rejection_ratio, "n_rejected / n_samples"],
                      ['last_accepted', self.last_accepted, "indicates if the last sample was accepted"],
                      ['sample_ratio', self.sample_ratio, "n_samples of the node / n_samples of the graph"],
                      ['last_threshold', self.last_threshold,
                       "threshold value (Metropolis Hastings) for the last sample"],
                      ['last_random_number', self.last_random_number, "random number for the last sample"],
                      ['last_seed', self.last_seed, "seed for the last sample"], ]
        for param in self.last_parameters.keys():
            table_data += [[param, self.last_parameters[param], "parameter of the last sampling method"]]
        table = ipy_table.make_table(table_data)
        table = ipy_table.apply_theme('basic_left')
        table = ipy_table.set_global_style(float_format="%3.3f")
        return table._repr_html_()


class SamplerState():
    """SamplerSate: stores the state of the Sampler, including information about 
    each node and global information. """

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.n_accepted = 0
        self.n_rejected = 0
        self.n_samples = 0
        self.rejection_ratio = 0.0
        self.last_node = None
        self.last_accepted = None
        self._nodes_state = {}

    def _has_node(self, nodename):
        return self._nodes_state.has_key(nodename)

    def get_n_nodes(self):
        return len(self._nodes_state)

    def get_state_node(self, nodename):
        if not self._has_node(nodename):
            raise ParameterError("There is no information regarding node '%s'" % nodename)
        return self._nodes_state[nodename]

    def new_sample(self, nodename, method, accepted, threshold, random_number, seed, parameters):
        # if new node name was previously unknown, create record for the node 
        if not self._has_node(nodename):
            self._nodes_state[nodename] = SamplerStateNode(nodename)

            # update node state
        node_state = self._nodes_state[nodename]
        node_state.new_sample(method, accepted, threshold, random_number, seed, parameters, self.n_samples)
        self._nodes_state[nodename] = node_state

        # update global sampler state
        self.n_samples += 1
        if accepted:
            self.n_accepted += 1
        else:
            self.n_rejected += 1
        self.last_node = nodename
        self.last_accepted = bool(accepted)
        self.rejection_ratio = self.n_rejected / (1.0 * self.n_samples)

    def to_dictionary(self):
        dictionary = {
            'n_nodes': self.get_n_nodes(),
            'n_accepted': self.n_accepted,
            'n_rejected': self.n_rejected,
            'n_samples': self.n_samples,
            'rejection_ratio': self.rejection_ratio,
            'last_accepted': self.last_accepted,
            'last_method': self.last_method}
        return dictionary

    def to_pickle(self):
        dictionary = self.to_dictionary()
        return pickle.dumps(dictionary)

    def _repr_html_(self):
        table_data = [['n_nodes', self.get_n_nodes(), "number of nodes recorded"],
                      ['n_samples', self.n_samples, "total number of samples"],
                      ['n_accepted', self.n_accepted, "number of samples accepted"],
                      ['n_rejected', self.n_rejected, "number of samples rejected"],
                      ['rejection_ratio', self.rejection_ratio, "n_rejected / n_samples"],
                      ['last_accepted', self.last_accepted, "indicates if the last sample was accepted"], ]
        table = ipy_table.make_table(table_data)
        table = ipy_table.apply_theme('basic_left')
        table = ipy_table.set_global_style(float_format="%3.3f")
        return table._repr_html_()


# Sampler 
class Sampler(object):
    """Sampler: samples from a ProbabilisticGraphicalModel. """

    def __init__(self, graph, optimization_only=False):
        self._nodes_compatible_sampling_methods = {}
        self._nodes_active_sampling_method = {}
        self._nodes_sampling_parameters = {}
        self._sampling_methods = []
        self.__load_sampling_methods()
        self.graph = None
        if graph is not None:
            self.attach_to_graph(graph, optimization_only)
        # By default, alternate nodes randomly:  
        self.set_sampling_strategy(RandomNodesStrategy)
        # By default, trace in RAM 
        self.attach_to_tracer(RamTracer())
        # Instantiate state:
        self.state = SamplerState()

    def _has_graph(self):
        return isinstance(self.graph, ProbabilisticGraphicalModel)

    def attach_to_graph(self, graph, optimization_only=False):
        if not isinstance(graph, ProbabilisticGraphicalModel):
            raise UnexpectedParameterType("graph must be an instance of ProbabilisticGraphicalModel. ")
        self.graph = graph
        # for each node, determine which sampling methods can be used  
        for node in self.graph.get_nodes():
            sampling_methods = []
            for sampling_method in self._sampling_methods:
                if sampling_method.can_be_used_with_node(node):
                    if not optimization_only:
                        sampling_methods.append(sampling_method)
                    else:
                        if sampling_method.is_optimizer():
                            sampling_methods.append(sampling_method)
            self._nodes_compatible_sampling_methods[name(node)] = sampling_methods
            if sampling_methods == []:
                print_important('None of the sampling methods can sample from node %s.\n' % name(node))
                # by default set the sampling method automatically for all nodes
        self.set_sampling_method_auto(optimization_only)
        self.__define_dynamic_methods()

    def attach_to_tracer(self, tracer):
        #        if not isinstance(tracer,Tracer):
        #            raise UnexpectedParameterType("tracer must be an instance of Tracer. ")
        self.tracer = tracer

    def set_node_sampling_method_auto(self, node, optimization_only=False):
        if not self.graph.has_node(node):
            raise ParameterError("The graph does not contain node '%s'." % name(node))
            # filter the methods if only optimization is requested
        if optimization_only:
            sampling_methods = []
            for sampling_method in self._nodes_compatible_sampling_methods[name(node)]:
                if sampling_method.is_optimization():
                    sampling_methods.append(sampling_method)
        else:
            sampling_methods = self._nodes_compatible_sampling_methods[name(node)]
        #        print_debug("The samplers compatible with are: %s\n"%name(self._nodes_compatible_sampling_methods[name(node)]) )
        # choose the best sampler 
        self._nodes_active_sampling_method[name(node)] = best_sampling_method(self.graph, node, sampling_methods)
        # use the default parameters for the sampling method 
        self._nodes_sampling_parameters[name(node)] = self._nodes_active_sampling_method[
            name(node)].default_parameters()
        #        print_debug("Selected sampling method '%s' for node '%s'. \n"%( name(self._nodes_active_sampling_method[name(node)]),name(node)) )
        return self._nodes_active_sampling_method[name(node)]

    def set_sampling_method_auto(self, optimization_only=False):
        for node in self.graph.get_nodes():
            self.set_node_sampling_method_auto(node, optimization_only)

    def get_sampling_method_node(self, node):
        return self._nodes_active_sampling_method[name(node)]

    def get_sampling_parameters_node(self, node):
        return self._nodes_sampling_parameters[name(node)]

    def set_node_sampling_method_manual(self, node, sampling_method_name, parameters=None):
        if not self.graph.has_node(node):
            raise ParameterError("The graph does not contain node '%s'." % name(node))
        if not sampling_method_name in map(name, self._nodes_compatible_sampling_methods[name(node)]):
            raise ParameterError(
                "Sampling method '%s' is not compatible with node '%s'." % (sampling_method_name, name(node)))
        self._nodes_active_sampling_method[name(node)] = self._nodes_compatible_sampling_methods[name(node)][
            map(name, self._nodes_compatible_sampling_methods[name(node)]).index(sampling_method_name)]
        if parameters is None:
            parameters = self._nodes_active_sampling_method[name(node)].default_parameters()
        self._nodes_sampling_parameters[name(node)] = parameters

    def sample_node(self, node, nsamples=1, trace=True, parameters=None, display_progress=True):
        node = name(node)
        # check if the node exists: 
        if not self.graph.has_node(node):
            raise ParameterError("The graph does not contain node '%s'." % node)
            # if node has given value, just return the value:
        node_is_given = self.graph.get_node(node).is_given()
        # check if a sampling strategy for the given node has been selected: 
        if not self._nodes_active_sampling_method.has_key(node):
            raise NotInitialized("The sampling strategy for node '%s' has not beel selected. " % node)
            # initialize progress bar:
        if display_progress:
            progress_bar = ProgressBar(height='6px', width='100%%', background_color=LIGHT_RED, foreground_color=RED)
        # sample: 
        if not node_is_given:
            active_sampling_method = self._nodes_active_sampling_method[node]
            for i in range(nsamples):
                # update progress bar
                if display_progress:
                    progress_bar.set_percentage(i * 100.0 / nsamples)
                sample = active_sampling_method.sample(self.graph, node, parameters)
                # update state of the graphical model: 
                self.graph.set_node_value(node, sample)
                # update sampler state
                self.state.new_sample(name(node), name(active_sampling_method), True, 0.2, 2.24, 234, {'delta': 0.01})
                # FIXME: fix line above: make the samplers return that information
                # trace
                if trace:
                    self.tracer.new_sample(name(node), sample)
        else:
            sample = self.graph.get_node_value(node)
            if trace:
                for i in range(nsamples):
                    # update state of the graphical model: 
                    self.graph.set_node_value(node, sample)
                    # update sampler state 
                    self.state.new_sample(name(node), "given", True, 0.2, 2.24, 234, {'delta': 0.01})
                    # trace 
                    self.tracer.new_sample(name(node), sample)
                    # update progress bar
                    if display_progress:
                        progress_bar.set_percentage(i * 100.0 / nsamples)
        if display_progress:
            progress_bar.set_percentage(100.0)
        return sample

    def sample(self, nsamples=1, trace=True, nsamples_per_node=1, display_progress=True):
        if display_progress:
            progress_bar = ProgressBar(height='6px', width='100%%', background_color=LIGHT_RED, foreground_color=RED)
        # alternate the nodes according to the sampling strategy: 
        for i in range(nsamples):
            if display_progress:
                progress_bar.set_percentage(i * 100.0 / nsamples)
            node = self._sampling_strategy.next_node()
            sample = self.sample_node(node, nsamples_per_node, trace, display_progress=False)
        if display_progress:
            progress_bar.set_percentage(100.0)
        return sample

    def set_sampling_strategy(self, sampling_strategy):
        if not issubclass(sampling_strategy, SamplingStrategy):
            raise UnexpectedParameterType("sampling_strategy must be a class of type SamplingStrategy. ")
        self._sampling_strategy = sampling_strategy(self.graph)

    def get_sampling_strategy(self):
        return self._sampling_strategy

    def get_state_node(self, nodename):
        return self.state.get_state_node(nodename)

    def __define_dynamic_methods(self):
        nodenames = self.graph.list_nodes()
        for nodename in nodenames:
            def func(nsamples=1, trace=True, parameters=None, nodename=nodename):
                return self.sample_node(nodename, nsamples, trace, parameters)

            setattr(self, "sample_" + nodename, func)

    def __load_sampling_methods(self):
        self._sampling_methods = []
        for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            if issubclass(obj, SamplingMethod):
                if not obj == SamplingMethod:
                    self._sampling_methods.append(obj())

    def _list_sapling_methods_names(self):
        l = []
        for m in self._sampling_methods:
            l.append(name(m))
        return l

    sampling_methods = property(_list_sapling_methods_names, None, None, "Empty doc. ")
