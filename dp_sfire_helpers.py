import networkx as nx
import numpy as np


def find_problems(success, defended, saved, burning_by_time, min_saved):
    union_defended = union_of_defended_dict(defended)
    for d in union_defended:
        if d not in saved:
            print(f'ERR: a vertex is in defended but not in saved')
            print(f'     {d} in {union_defended}, not in {saved}')

    union_fire = set()
    for f in burning_by_time:
        union_fire.update(burning_by_time[f])
    for f in union_fire:
        if f in union_defended:
            if f in union_defended:
                raise Exception('ERR: a burning vertex is also defended\n     '
                                '{f} in {union_fire} AND {union_defended}')
            elif f in saved:
                raise Exception('ERR: a burning vertex is also saved\n     {f} in {union_fire} AND {saved}')

    num_saved = len(saved)
    num_defended = len(union_defended)
    num_burned = len(union_fire)
    if num_saved < num_defended:
        raise Exception('ERR: number of saved vertices smaller than number of defended vertices\n    '
                        'SAVED: len({saved})={len(saved)}, DEFENDED: len({union_defended})={len(union_defended)}')
    # assert success == (len(saved) >= min_saved)


def union_of_defended_dict(d):
    union = set()
    for i in d:
        for key in d[i]:
            union.update(d[i][key])
    return list(union)


def is_chordal(g):
    if g.number_of_nodes() < 4:
        return True
    # Check for self-loops
    if any(v in g.neighbors(v) for v in g.nodes()):
        return False
    # Use the original is_chordal() function from NetworkX
    return nx.is_chordal(g)


def create_clique_tree(graph, root, p=False):
    if p:
        print('*** START: create clique tree ***')
    cliques = list(nx.find_cliques(graph))
    if p:
        print('List of cliques', cliques)

    num_cliques = len(cliques)
    clique_tree = np.zeros((num_cliques, num_cliques), dtype=int)  # Adjacency matrix

    # Create a dictionary to store vertices and their corresponding cliques
    vertex_to_cliques = {v: set() for v in graph.nodes()}
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_to_cliques[v].add(i)
    if p:
        print('vertex-clique dictionary:', vertex_to_cliques)

    # Create the clique tree using the more efficient algorithm
    for i in range(num_cliques):
        for j in range(i + 1, num_cliques):
            if any(vertex_to_cliques[v].intersection({i, j}) == {i, j} for v in cliques[i]):
                clique_tree[i][j] = clique_tree[j][i] = 1
    if p:
        print("Clique tree adjacency matrix:\n", clique_tree)
        print("cliques", cliques)

    if p:
        print('*** END: create clique tree ***')
    root_clique = [root]
    for K in cliques:
        if root in K:
            root_clique = K
            break
    return clique_tree, cliques, root_clique
