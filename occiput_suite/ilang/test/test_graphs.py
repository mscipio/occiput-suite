# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 

from __future__ import absolute_import, print_function

import unittest
from .. import Graphs
from .. import exceptions
from .. import verbose

verbose.set_verbose_low()


class TestSequenceProbabilisticGraphicalModel(unittest.TestCase):
    """Sequence of tests for ProbabilisticGraphicalModel class"""

    def setUp(self):
        pass

    def test_ProbabilisticGraphicalModel_nodes(self):
        """Test whether ProbabilisticGraphicalModel correctly manages inserting and deleting nodes"""
        dag = Graphs.ProbabilisticGraphicalModel()
        # 1) add node instance 
        node1 = Graphs.Node('node1', value=0, given=False)
        dag.add_nodes(node1)
        # 2) add node by name
        self.assertEqual(node1.name, dag.get_node('node1').get_name())
        dag.add_nodes('node2')
        node2 = dag.get_node('node2')
        self.assertTrue(node2 in dag.get_nodes())
        # 3) check if exception is thrown in case of duplicate node names 
        self.assertRaises(exceptions.ParameterError, dag.add_nodes, 'node1')
        self.assertRaises(exceptions.ParameterError, dag.add_nodes, node2)
        # 4) remove node 
        self.assertRaises(exceptions.ParameterError, dag.remove_node, 'node3')
        dag.remove_node('node2')
        self.assertEqual(dag.get_nodes(), [node1, ])
        # 5) add multiple nodes 
        dag.add_nodes(['node3', 'node4'])
        self.assertTrue('node3' in dag.get_nodes_names() and 'node4' in dag.get_nodes_names())

    def test_ProbabilisticGraphicalModel_dependencies(self):
        """Test whether ProbabilisticGraphicalModel correctly manages attaching and detaching dependencies"""
        dag = Graphs.ProbabilisticGraphicalModel()
        # 1) attach a dependence 

        # 2) detach a dependence 

        # 3) connect a dependence 

        # 4) fully connected property 

        # 5) complete graph property 

    def test_ProbabilisticGraphicalModel_inference(self):
        """Test methods of ProbabilisticGraphicalModel related to inference"""
        # 1) test extraction of Markov blanket
        pass
        # 2) test d-separation property 

        # 3) test parents, children 

    def test_ProbabilisticGraphicalModel_export(self):
        """Test whether ProbabilisticGraphicalModel correctly manages exporting graph"""
        # 1) export dictionary
        pass


if __name__ == '__main__':
    unittest.main()
