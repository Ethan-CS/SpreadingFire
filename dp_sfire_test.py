import unittest

from dp_sfire import *
from dp_sfire_helpers import union_of_defended_dict, create_clique_tree


def get_all_defended(defended):
    all_defended = set()
    for i in defended.keys():
        all_defended.update(set(defended[i]['direct']).union(set(defended[i]['spread'])))
    return all_defended


class TestFirefightingAlgorithm(unittest.TestCase):
    def test_create_clique_tree_small(self):
        # Test case 1: Simple chordal graph
        G1 = nx.Graph([(0, 1), (1, 2), (2, 0)])
        clique_tree, cliques, _ = create_clique_tree(G1, 0)
        self.assertEqual(len(cliques), 1)
        self.assertEqual(set(cliques[0]), {0, 1, 2})
        self.assertEqual(clique_tree.shape, (1, 1))

    def test_create_clique_tree_larger(self):
        # Test case 2: Larger chordal graph
        G2 = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 4), (4, 5)])
        clique_tree, cliques, _ = create_clique_tree(G2, 0)
        cliques_as_sets = [set(K) for K in cliques]
        # cliques_as_sets = {set(K) for K in cliques}
        self.assertEqual(len(cliques), 4)
        msg = f"cliques: {cliques}"
        self.assertTrue({0, 1, 2} in cliques_as_sets, msg)
        self.assertTrue({0, 2, 3} in cliques_as_sets, msg)
        self.assertTrue({3, 4} in cliques_as_sets, msg)
        self.assertTrue({4, 5} in cliques_as_sets, msg)
        self.assertEqual(clique_tree.shape, (4, 4))

    def test_s_fire_lollipop_graph(self):
        G = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
        root = 0
        min_save = 2
        budget = 1

        success, defended, saved, _ = s_fire(G, root, min_save, budget)

        success_dec, defended_dec, saved_dec, _ = s_fire(G, root, min_save, budget, problem='DEC', p=True)
        self.assertTrue(success, f'saved: {saved_dec}')
        self.assertTrue(success, f'saved: {saved} {[f"{d}: {defended[d]}" for d in defended]}')
        self.assertEqual(len(defended), 2, defended)
        self.assertTrue(2 and 3 in get_all_defended(defended))  # 2, spreads to 3
        self.assertEqual(set(saved), {2, 3})
        self.assertEqual(set(G.nodes()) - set(saved), {0, 1})  # 0 and 1 should be burning

    def test_s_fire_larger_graph(self):
        G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2),  # square with a diagonal
                      (3, 4),  # plus an edge to ...
                      (4, 5), (5, 6), (6, 4)])  # a triangle
        root = 0
        min_save = 5
        budget = 1

        success, defended, saved, _ = s_fire(G, root, min_save, budget)
        all_defended = get_all_defended(defended)
        msg = f'success={success},\ndefence={defended},\nall defended={all_defended},\nsaved={saved}'
        self.assertFalse(success, f'Should *not* be able to save 5 vertices\n{msg}')
        self.assertEqual(len(all_defended), 4, f'Should defend 2 and spread to 2 i n2 time steps\n{msg}')
        self.assertTrue(3 in all_defended, f'Vertex 3 should be defended\n{msg}')
        self.assertEqual(set(saved), {3, 4, 5, 6}, msg)
        self.assertEqual(set(G.nodes()) - set(saved), {0, 1, 2}, f'0, 1, and 2 should all be burning\n{msg}')

    def test_s_fire_min_save_edge_cases(self):
        G = nx.Graph([(0, 1), (1, 2), (2, 0)])
        root = 0

        # Test when min_save is 0
        success, defended, saved, _ = s_fire(G, 0, 0, 1)
        self.assertTrue(success)
        self.assertEqual(len(saved), 2)  # Should save 2 vertices

        # Test when min_save is greater than number of vertices
        success, defended, saved, burning_by_time = s_fire(G, 0, 4, 1)
        self.assertFalse(success)
        self.assertTrue(len(saved) < 4)

    # def test_multiple_optimal_strategies(self):
    #     # Graph where defending either vertex 1 or 2 leads to the same outcome
    #     G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (2, 4)]) # TODO not chordal
    #     root = 0
    #     min_save = 4
    #     budget = 1
    #
    #     success, defended, saved = s_fire(G, root, min_save, budget)
    #
    #     self.assertTrue(success)
    #     self.assertEqual(len(saved), 5)
    #     self.assertTrue(defended[0] in [1, 2], "Optimal strategy should defend either 1 or 2")

    def test_max_vertices_saved(self):
        # Simple chordal graph
        G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        #     0---1
        #     |\  |
        #     | \ |
        #     |  \|
        #     3---2
        root = 0
        budget = 1

        # Test with different min_save values
        for min_save in range(1, 5):
            success, defended, saved, burning_by_time = s_fire(G, root, min_save, budget)
            msg = f'success: {success}, defended: {defended}, saved: {saved}'
            if min_save == 1:
                self.assertTrue(success, f"Should succeed for min_save={min_save}\n{msg}")
            else:
                self.assertFalse(success, f"Should fail for min_save={min_save}\n{msg}")
            self.assertEqual(len(saved), 1, f"Should save exactly 3 vertices for min_save={min_save}\n{msg}")

    # def test_various_budgets(self):
    #     G = nx.Graph([(0, 1), (0, 2), (0, 5), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5)]) # TODO: not chordal!
    #     root = 0
    #     min_save = 5
    #
    #     # Test with budget 1
    #     success1, defended1, saved1 = s_fire(G, root, min_save, 1)
    #     self.assertTrue(success1)
    #     self.assertEqual(len(saved1), 5)
    #
    #     # Test with budget 2
    #     success2, defended2, saved2 = s_fire(G, root, min_save, 2)
    #     self.assertTrue(success2)
    #     self.assertEqual(len(saved2), 6)
    #
    #     # Test with budget 3 (should save all vertices)
    #     success3, defended3, saved3 = s_fire(G, root, min_save, 3)
    #     self.assertTrue(success3)
    #     self.assertEqual(len(saved3), len(G))

    def test_large_chordal_graph(self):
        # Create a larger chordal graph
        G = nx.Graph()
        G.add_edges_from([
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),  # Outer cycle
            (0, 2), (2, 4), (4, 6)  # Chords
        ])
        root = 2
        min_save = 4  # expecting to save 0, 1 or 3 first

        success, defended, saved, burning_by_time = s_fire(G, root, min_save)
        print(burning_by_time)
        msg = f"\nsuccess: {success}, \ndefended: {defended}, \nsaved: {saved}"

        self.assertTrue(success, f"Not successful. Output: {msg}")
        self.assertGreaterEqual(len(saved), min_save, msg)
        self.assertEqual(len(union_of_defended_dict(defended)), min_save, msg)

    def test_edge_cases(self):
        # Single vertex graph
        G1 = nx.Graph()
        G1.add_node(0)
        success, defended, saved, _ = s_fire(G1, 0, 1, 1)
        self.assertFalse(success)
        self.assertEqual(len(saved), 0)

        # Two vertex graph
        G2 = nx.Graph([(0, 1)])
        success, defended, saved, _ = s_fire(G2, 0, 1, 1)
        self.assertTrue(success)
        self.assertEqual(len(saved), 1)

        # Graph with isolated vertices
        G3 = nx.Graph()
        G3.add_node(0)
        G3.add_node(1)
        G3.add_node(2)
        success, defended, saved, _ = s_fire(G3, 0, 2, 1)
        self.assertTrue(success)
        self.assertEqual(len(saved), 2)


if __name__ == '__main__':
    unittest.main()
