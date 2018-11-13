# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# 20 Oct 2013, Helsinki 
# Harvard University, Martinos Center for Biomedical Imaging
# Dec 2013, Boston, MA, USA 

from __future__ import absolute_import, print_function

__all__ = ['Model', 'MultivariateGaussian', 'Poisson', 'Smoothness', 'Library']

from .Graphs import Dependence
from .exceptions import *
from .verbose import *

from ..DisplayNode import DisplayNode
import numpy


class Model(Dependence, object):
    """A Model is a Dependence with name. """

    def __init__(self, name):
        Dependence.__init__(self, name)


class MultivariateGaussian(Model):
    variables = {'x': 'continuous', 'mu': 'continuous', 'cov': 'continuous'}
    dependencies = [['mu', 'x', 'directed'], ['cov', 'x', 'directed']]
    preferred_samplers = {'x': ['HamiltonianMCMC']}

    # init
    def init(self):
        pass

        # expose to sampler:

    def log_conditional_probability_x(self, x):
        hessian = self._compute_hessian()
        mu = self.get_value('mu')
        return -.5 * numpy.dot(numpy.dot((x - mu), hessian), (x - mu).T)

    def log_conditional_probability_gradient_x(self, x):
        hessian = self._compute_hessian()
        mu = self.get_value('mu')
        return -.5 * numpy.dot((x - mu), hessian + hessian.T)

    def log_conditional_probability_hessian_x(self, x):
        hessian = self._compute_hessian()
        return hessian

    def log_conditional_probability_mu(self, mu):
        # FIXME: implement
        return 0

    def log_conditional_probability_gradient_mu(self, mu):
        # FIXME: implement
        return 0

    def sample_conditional_probability_x(self):
        mu = self.get_value('mu')
        cov = self.get_value('cov')
        return numpy.random.multivariate_normal(mu.reshape(mu.size), cov)

        # miscellaneous:

    def _compute_hessian(self):
        cov = self.get_value('cov')
        self._hessian = numpy.linalg.inv(cov)
        return self._hessian


class Poisson(Model):
    variables = {'lambda': 'continuous', 'alpha': 'continuous', 'z': 'discrete'}
    dependencies = [['lambda', 'z', 'directed'], ['alpha', 'z', 'directed']]
    preferred_samplers = {'lambda': ['ExpectationMaximization'], 'alpha': ['ExpectationMaximization']}

    def init(self):
        pass

    def log_conditional_probability_lambda(self, Lambda):
        return 0

    def log_conditional_probability_gradient_lambda(self, Lambda):
        return 0

    def log_conditional_probability_alpha(self, alpha):
        return 0

    def log_conditional_probability_gradient_alpha(self, alpha):
        return 0

    def sample_conditional_probability_counts(self, counts):
        return 0


class Smoothness(Model):
    variables = {'x': 'continuous', 'beta': 'continuous'}
    dependencies = [['beta', 'x', 'directed']]

    def init(self):
        pass

    def log_conditional_probability_x(self, x):
        return 0

    def log_conditional_probability_gradient_x(self, x):
        return 0

    def log_conditional_probability_beta(self, beta):
        return 0

    def log_conditional_probability_gradient_beta(self, beta):
        return 0


class SSD_Transformation(Model):
    variables = {'source': 'continuous', 'target': 'continuous', 'transformation': 'continuous', 'sigma': 'continuous'}
    dependencies = [['source', 'target', 'directed'], ['transformation', 'target', 'directed'],
                    ['sigma', 'target', 'directed']]

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        Model.__init__(self, name)

    def log_conditional_probability_transformation(self, transformation):
        source = self.get_value('source')
        target = self.get_value('target')
        sigma = self.get_value('sigma')
        log_p = 0.0
        return log_p

    def log_conditional_probability_gradient_transformation(self, transformation):
        source = self.get_value('source')
        target = self.get_value('target')
        sigma = self.get_value('sigma')
        gradient = numpy.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return gradient

    def sample_conditional_probability_target(self):
        return 0

    def init(self):
        pass


class Library():
    def __init__(self):
        self.models = []
        self._make_inventory()

    def list_models(self):
        return self.models

    def _make_inventory(self):
        models = []
        import inspect, sys
        for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            if issubclass(obj, Model):
                if not obj == Model:
                    models.append(obj)
        self.models = models
        return models

    def export_dictionary(self):
        nodes = []
        links = []
        for model in self.list_models():
            d = model.export_dictionary()
            model_nodes = d['nodes']
            model_links = d['links']
            for node in model_nodes:
                node['name'] = model.get_name() + ': ' + node['name']
            for link in model_links:
                link['source'] = model.get_name() + ': ' + link['source']
                link['target'] = model.get_name() + ': ' + link['target']
            nodes += model_nodes
            links += model_links
        dictionary = {'nodes': nodes, 'links': links}
        return dictionary

    def export_json(self):
        from json import dumps
        return dumps(self.export_dictionary())

    def display_in_browser(self):
        """Displays the graph in a web browser. """
        display_node = DisplayNode()
        display_node.display_in_browser('graph', self.export_dictionary())

    def _repr_html_(self):
        display_node = DisplayNode()
        return display_node.display('graph', self.export_dictionary())._repr_html_()
