import itertools
import time

import networkx as nx
import pandas as pd

from dp_sfire_helpers import create_clique_tree


def calculate_saved_vertices(clique, defend, burning_vertices, defended_vertices):
    # Count the defended vertices
    saved = defend
    to_defend = []

    # Check how many undefended vertices in the clique are not adjacent to burning vertices
    for vertex in clique:
        if vertex not in burning_vertices and any(adj in burning_vertices for adj in nx.neighbors(graph, vertex)):
            continue  # This vertex is adjacent to a burning vertex, cannot be saved
        if vertex not in burning_vertices and vertex not in defended_vertices:  # It's undefended and not burning
            saved += 1  # This vertex can be saved
            to_defend.append(vertex)
    print(f"save {saved} by defending {to_defend}")
    return saved, to_defend


def dp(graph, clique_tree, cliques, budget, i, time, burning_vertices, defended_vertices, memo, dp_table, min_save, p=False):
    p and print(f"DP FUNCTION CALL: Clique {i}, Time {time}, Burning: {burning_vertices}, Defended: {defended_vertices}")
    if i > len(cliques)-1 or i >= min_save+1:
        return 0

    max_saved = len(defended_vertices)
    if len(defended_vertices) + len(burning_vertices) == graph.number_of_nodes():
        return len(defended_vertices)

    p and print(f"{' .' * time}considering clique {cliques[i]} with budget {budget} at time {i}, "
              f"defended {defended_vertices} burned {burning_vertices}")

    if (i, budget, time) in memo:
        p and print(f"{' .' * (time+1)}worked {(i, budget, time)} out already")
        return memo[(i, budget, time)]

    defensible_vertices = [v for v in cliques[i] if v not in burning_vertices and v not in defended_vertices]
    p and print(f"{' .' * (time+1)}defensible vertices:", defensible_vertices)

    could_defend = []
    for length in range(min(budget, len(cliques[i]))+1):
        for subset in itertools.combinations(defensible_vertices, length):
            if len(subset) > 0:
                could_defend.append(set(subset))

    if not could_defend:
        p and print(f"{' .' * time}could not defend anything")
        if i + 1 <= max_saved + 1:
            return dp(graph, clique_tree, cliques, budget, i + 1, time, burning_vertices, defended_vertices, memo, dp_table,
                  min_save)
        else:
            return 0

    p and print(f"{' .' * (time+1)}possible defences this turn:", could_defend)

    for try_defend in could_defend:
        p and print(f'{" ." * (time+2)}going to try defending', try_defend, 'of', could_defend, 'with', defended_vertices,
                  'already defended')

        max_saved, new_burning, new_defended = do_spreading(burning_vertices, defended_vertices, graph, max_saved,
                                                            try_defend, p)

        to_increase = 0
        if budget - 1 > 0:
            to_increase = dp(graph, clique_tree, cliques, budget - 1, i, time + 1, new_burning, new_defended, memo,
                             dp_table, min_save)

        saved_in_child = 0
        for n in nx.Graph(clique_tree).neighbors(i):
            if n > i:  # Only consider child cliques
                new_saved_in_child = dp(graph, clique_tree, cliques, budget, n, time + 1, new_burning, new_defended, memo,
                                        dp_table, min_save)
                if new_saved_in_child > saved_in_child:
                    saved_in_child = new_saved_in_child
                    p and print(f'{" ." * (time+2)}saving in child', n, 'is better')
        p and print(f'{" ." * (time+1)}to_increase = {to_increase}, saved_in_child = {saved_in_child}, max_saved = {max_saved}')
        to_increase = max(to_increase, saved_in_child)
        max_saved = max(to_increase, saved_in_child, max_saved)
        # if max_saved >= min_save:
        #     return min_save

    print('i', i, 'dp', dp_table)
    memo[(i, budget, time)] = max_saved
    dp_table[i][budget][time] = max_saved
    p and print(f'{" ." * (time+1)}updated dp: dp[{i}][{budget}][{time}]={max_saved}')
    for d in dp_table:
        print(d)
    return max_saved


def do_spreading(burning_vertices, defended_vertices, graph, max_saved, try_defend, p=False):
    new_defended = set(defended_vertices)
    new_defended.update(try_defend)

    new_burning = set(burning_vertices)
    for burning in burning_vertices:
        for neighbour in graph.neighbors(burning):
            if neighbour not in new_defended and neighbour not in burning_vertices:
                new_burning.add(neighbour)

    spread = []
    for defended in new_defended:
        for neighbour in graph.neighbors(defended):
            if neighbour not in new_burning and neighbour not in new_defended:
                spread.append(neighbour)
                max_saved += 1
    new_defended.update(spread)
    p and print('new burning:', new_burning, ' all burning now:', burning_vertices, '\n',
                'new defended:', new_defended, 'all defended now:', defended_vertices, '\n',
                'max_saved now:', max_saved)
    return max_saved, new_burning, new_defended


def SFire(G, r, b, k, p=False):
    # Get maximal cliques using networkx
    start_time_tree = time.time()
    clique_tree, cliques, root_clique = create_clique_tree(G, r, p)
    end_time_tree = time.time()
    p and print(cliques)
    print(f'  time to get clique tree: {end_time_tree - start_time_tree}')
    memo = {}

    # Initialize DP table and defense count - [clique index][budget][time]
    dp_table = [[[0 for _ in range(max(k+1, 2))] for _ in range(b+1)] for _ in range(len(cliques))]

    # Start with the burning vertex (the root)
    burning_vertices = {r}
    defended_vertices = set()

    # Compute the maximum saved vertices
    start_time_dp = time.time()
    max_saved_vertices = dp(graph=G, clique_tree=clique_tree, cliques=cliques, budget=b, i=0, time=0,
                            burning_vertices={r}, defended_vertices=set(), memo=memo,
                            dp_table=dp_table, min_save=k, p=p)
    end_time_dp = time.time()
    print(f'  time to run DP program: {end_time_dp - start_time_dp}')

    # Now extract the strategy
    # strategy = extract_strategy(G, dp_table, cliques, b, k)
    # if strategy:
    #     print("Optimal strategy found:", strategy)
    # else:
    #     print("No valid strategy found to save at least", k, "vertices.")

    return max_saved_vertices >= k


def extract_strategy(graph, dp_table, cliques, b, k):
    """
    Extracts the strategy from the DP table.

    Parameters:
    - graph: The original graph (for adjacency information).
    - dp_table: The DP table containing the maximum saved vertices.
    - cliques: The list of cliques in the graph.
    - b: The initial defense budget.
    - k: The target number of saved vertices.

    Returns:
    - A list of tuples indicating the defended vertices for each clique, or None if no valid strategy exists.
    """
    strategy = []
    current_budget = b
    i = 0  # Start from the first clique

    # Iterate through the cliques
    while i < len(cliques):
        # If we cannot save at least k vertices, return None
        if dp_table[i][current_budget] < k:
            return None  # No valid strategy found

        clique_size = len(cliques[i])
        found = False

        # Determine how many vertices were defended in this clique
        for defend in range(min(clique_size, current_budget) + 1):
            # Calculate the burning vertices after this defense
            burning_vertices_next = set(cliques[i])  # All vertices in the current clique will burn if not defended

            # Check if this defense leads to the same saved vertices count
            if i == 0:  # First clique special case
                previous_saved = 0  # There are no previous cliques
            else:
                previous_saved = dp_table[i - 1][current_budget - defend] if current_budget - defend >= 0 else 0

            saved, _ = calculate_saved_vertices(cliques[i], defend, burning_vertices_next, strategy)

            # Check if the current state matches the DP table
            if dp_table[i][current_budget] == previous_saved + saved:
                strategy.append((i, defend))  # Store the clique index and the number of defenses
                current_budget -= defend  # Decrease the budget
                found = True
                break

        if not found:
            return None  # No valid defense found for this clique

        i += 1  # Move to the next clique

    return strategy


def run_example(graph, root, min_save, budget, expectation, sure=False, p=False):
    start_time = time.time()
    print('i =', min_save, 'in run_example')
    decision = SFire(G=graph, r=root, k=min_save, b=budget, p=p)
    end_time = time.time()

    elapsed_time = end_time - start_time

    if p:
        print(f"Success: {decision} (expected {expectation})")
        print(f'We expected success: {expectation} {"but we are not sure!" if sure else ""}')
        print(f"Time elapsed for SFire: {elapsed_time:.6f} seconds")

    if decision != expectation:
        raise Exception("Expected", expectation, "got", decision)
    else:
        if p:
            print(f"Good news! Expectation ({expectation}) matched decision ({decision})")
    if p:
        print()
    return decision


if __name__ == "__main__":
    # print(" *** Example 1: Small chordal graph ***")
    # print("""
    # 0 - 1
    # | /
    # 2 - 3 - 4
    # We expect: {0: burning [0], defend [2], fire to [1], defence to [3]; fire contained, saved [4]}
    # burning: [0, 1], defended directly: [2], indirectly: [3], all saved: [2, 3, 4]
    # """)
    # G1 = nx.Graph()
    # G1.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
    #
    # run_example(G1, root=0, min_save=3, budget=1, expectation=True)
    #
    # run_example(G1, root=0, min_save=4, budget=1, expectation=False)


    print(" *** Example 2: Larger chordal graph ***")
    print("""
    0 - 1
    | \\ |
    3 - 2
    |
    4 - 5
     \\ /
      6
     We expect: {0: defend 3, fire to 1 and 2, defence to 4, fire can no longer spread, 5 and 6 saved}
       i.e. burned: [0, 1, 2], defended directly: [3], defence spread: [4], saved: [5, 6]
     """)

    G2 = nx.Graph()
    G2.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0), (0, 2),  # Chordal cycle
        (3, 4), (4, 5), (5, 6), (6, 4)  # Rest of the graph
    ])
    run_example(G2, root=0, min_save=4, budget=1, expectation=True, p=True)

    run_example(G2, root=0, min_save=5, budget=1, expectation=False)

    print(" *** Example 3 ***")
    print("""
        0 - 1 - 2
        | \\ | / |
        3 - 4 - 5
         \\ /
          6
        We expect: {0: defend [4], fire to [1, 3], defence to [5, 6], fire can no longer spread, also save [2]}
          i.e. burned: [0, 1, 3], defended directly: [4], defence spread: [5, 6], saved: [2, 5]
        """)

    G3 = nx.Graph()
    G3.add_edges_from([
        (0, 1), (1, 2), (2, 5), (5, 4), (4, 3), (3, 0),  # Outer cycle
        (0, 4), (1, 4), (2, 4),  # Chords to make it chordal
        (3, 6), (4, 6)  # Connection to vertex 6
    ])

    run_example(G3, root=0, min_save=3, budget=1, expectation=True)

    run_example(G3, root=0, min_save=4, budget=1, expectation=True)

    run_example(G3, root=0, min_save=5, budget=1, expectation=True)

    run_example(G3, root=0, min_save=6, budget=1, expectation=False)

    print(" *** Example 4: Chordal graph with multiple cliques ***")
    print("""
        0 - 1 - 2
        | \\ | / |
        4 - 3 - 5
        | \\ |
        6 - 7
        
        0 burning
         - choose to defend 3
         - fire to 1, 4
         defence to 2, 5, 7
        [0, 1, 4] burning, [2, 3, 5, 7] defended
         - choose to defend 6
        END - SAVED [2, 3, 5, 6, 7], BURNED [0, 1, 4]
        """)
    G1 = nx.Graph()
    G1.add_edges_from([
        (0, 1), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 5),
        (3, 4), (3, 5), (3, 7),

    ])

    run_example(G1, root=0, min_save=4, budget=1, expectation=True)
    run_example(G1, root=0, min_save=5, budget=1, expectation=True)
    run_example(G1, root=0, min_save=6, budget=1, expectation=False)

    print(" *** Example 2: Larger chordal graph with complex structure ***")
    print("""
        0 - 1 - 2
        | \\ | / |
        4 - 3 - 5
        |       |
        6 - 7   8
        
        0 burning
         - choose to defend
        """)

    G2 = nx.Graph()
    G2.add_edges_from([
        (0, 1), (1, 2), (2, 5), (5, 3), (3, 4), (4, 6), (6, 7), (7, 8), (4, 1), (3, 1)
    ])

    run_example(G2, root=0, min_save=4, budget=1, expectation=True, p=True)
    run_example(G2, root=0, min_save=5, budget=1, expectation=False)





