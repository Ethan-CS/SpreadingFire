import networkx as nx
import numpy as np
import pandas as pd

from dp_sfire import reconstruct_defended_vertices
from dp_sfire_helpers import is_chordal, create_clique_tree, find_problems

# INFO works better than original `dp_sfire` but still is not recursive


def s_fire_decision_v2(graph, root, min_save, budget=1, p=False):
    if not is_chordal(graph):
        raise Exception('Graph is not chordal!')

    clique_tree, cliques, root_clique = create_clique_tree(graph, root, root)

    dp_optimal = {}
    dp = pd.DataFrame(columns=['clique', 'fire state', 'time', 'value'])
    burning = {root}

    # Process cliques in reverse order (bottom-up)
    for clique_index in range(len(cliques) - 1, -1, -1):
        child_indices = np.where(clique_tree[clique_index] == 1)[0]
        if len(child_indices) == 0:  # This is a leaf clique, move onto next
            continue

        for time_left in range(min_save + 1):
            for fire_state in [False, True]:
                clique_considering = set(cliques[clique_index])

                can_defend = []
                child_fire_state = False
                if not fire_state:
                    for v in clique_considering:
                        if v not in burning:
                            can_defend.append(v)
                        else:
                            child_fire_state = True
                    # Modified upper bound calculation
                    upper_bound = min(time_left * budget, len(can_defend)) + 1
                else:
                    upper_bound = min(budget, len(clique_considering)) + 1

                running_max_defended = 0
                best_k = 0
                for k in range(upper_bound):
                    defended_here = k
                    defended_children = sum(
                        dp_optimal.get((child_index, child_fire_state, max(0, time_left - 1)), 0)
                        for child_index in child_indices)
                    total_defended = defended_here + defended_children
                    if total_defended > running_max_defended:
                        running_max_defended = total_defended
                        best_k = k

                dp_optimal[(clique_index, fire_state, time_left)] = running_max_defended
                dp.loc[len(dp)] = {
                    'clique': clique_index,
                    'fire state': fire_state,
                    'time': min_save + 1 - time_left,
                    'value': running_max_defended
                    }

                # Update burning set
                if fire_state and best_k == 0:
                    burning.update(clique_considering)
    defended_vertices, burning_by_time, saved_vertices = (
        reconstruct_defended_vertices(graph=graph, cliques=cliques, min_save=min_save, root=root,
                                      dp_optimisation=dp_optimal, budget=budget, p=p))

    possible_wins = pd.DataFrame(columns=dp.columns)
    print(dp)
    decision = False
    print('begin possible wins')
    for i, row in dp.iterrows():
        if row['clique'] == cliques.index(root_clique) and row['fire state'] and row['value'] >= min_save:
            possible_wins.loc[len(possible_wins)] = row
            decision = True
    print(possible_wins)
    print('end')
    find_problems(decision, defended_vertices, saved_vertices, burning_by_time, min_save)
    for d in dp_optimal:
        print(d, dp_optimal[d])
    return decision, dp, defended_vertices, burning_by_time, saved_vertices


def s_fire_decision_v2_recur(graph, root, min_save, cliques, clique_tree, budget=1, time=0, burning=None, defended=None,
                             dp=None, p=False):
    """
    Recursively computes the optimal strategy for the Spreading Fire problem.

    Parameters:
    - graph: The graph representing the problem.
    - root: The root vertex from which the fire starts.
    - min_save: Minimum number of vertices that need to be saved.
    - budget: Maximum number of vertices that can be defended in one time step.
    - time: Current time step in the recursion.
    - burning: Set of currently burning vertices.
    - dp: DataFrame for storing decision-making information.
    - p: Flag for printing debug information.

    Returns:
    - success: Boolean indicating whether the target number of vertices was saved.
    - dp: Updated decision DataFrame.
    - defended_vertices: Set of defended vertices.
    - burning_by_time: Dictionary of burning vertices by time step.
    - saved_vertices: List of vertices that are saved.
    """
    dots = " " * time
    print(dots)
    print(f"{dots}Processing time {time} with burning vertices {burning}")

    if burning is None:
        burning = {root}  # Initialize the burning set if not provided

    # Base case: Check if we have saved enough vertices
    if time >= min_save:
        print(f"Base case reached at time {time}")
        return dp, set(), {}, {}

    # Create the clique tree and cliques if dp is not initialized
    if dp is None:
        dp = pd.DataFrame(columns=['clique', 'fire state', 'time', 'value'])

    # Process each clique in reverse order (bottom-up)
    for clique_index in range(len(cliques) - 1, -1, -1):
        print(f"{dots}..Processing clique with index {clique_index}, i.e. {cliques[clique_index]}")
        child_indices = np.flatnonzero(clique_tree[clique_index] == 1)
        if len(child_indices) == 0:  # Skip leaf cliques
            continue

        for time_left in range(min_save + 1):
            for fire_state in [False, True]:
                clique_considering = set(cliques[clique_index])

                # Determine defensible vertices
                can_defend = [v for v in clique_considering if v not in burning]
                child_fire_state = any(v in burning for v in clique_considering)

                # Calculate upper bound for defending vertices
                if not fire_state:
                    upper_bound = min(time_left * budget, len(can_defend)) + 1
                else:
                    upper_bound = min(budget, len(clique_considering)) + 1

                running_max_defended = 0  # Track the maximum defended vertices
                defended_here = 0
                for k in range(upper_bound):
                    defended_here = min(budget, len([v for v in clique_considering if v not in burning]))
                    defended_children = sum(
                        dp.loc[
                            (dp['clique'] == child_index) &
                            (dp['fire state'] == child_fire_state) &
                            (dp['time'] == max(0, time_left - 1)),
                            'value'
                        ].sum() for child_index in child_indices
                    )
                    running_max_defended = max(running_max_defended, defended_here + defended_children)

                # Store results in the decision table
                dp.loc[len(dp)] = {
                    'clique': clique_index,
                    'fire state': fire_state,
                    'time': min_save + 1 - time_left,
                    'value': running_max_defended
                }

                # Update burning set if no vertices are defended
                if fire_state and defended_here == 0:
                    burning.update(clique_considering)

    # Recursive call for the next time step
    return s_fire_decision_v2_recur(graph, root, min_save, cliques, clique_tree, budget, time + 1, burning, defended,
                                    dp, p)


if __name__ == "__main__":
    def run_example(G, root, min_save, budget, expectation, sure=False):
        clique_tree, cliques, root_clique = create_clique_tree(G, root, root)
        decision, dp, defended_vertices, burning_by_time, saved_vertices = s_fire_decision_v2(G, root, min_save, budget)

        print(f"Success: {decision} (expected {expectation}), saved {len(saved_vertices)} vertices and needed to save at least {min_save}")

        # print(f'We expected success: {expectation} {"but we are not sure!" if sure else ""}')
        # if decision != expectation:
        #     raise Exception("Expected", expectation, "got", decision)
        # else:
        #     print(f"Good news! Expectation ({expectation}) matched decision ({decision})")
        # print()


    print(" *** Example 1: Small chordal graph ***")
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
    run_example(G1, root=0, min_save=3, budget=1, expectation=True)

    print(" *** Example 2: Larger chordal graph ***")
    G2 = nx.Graph()
    G2.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0), (0, 2),  # Chordal cycle
        (3, 4), (4, 5), (5, 6), (6, 4)  # Rest of the graph
    ])
    run_example(G2, root=0, min_save=5, budget=1, expectation=False)
    #
    # print(" *** Example 3: Complete graph ***")
    # run_example(nx.complete_graph(6), root=0, min_save=4, budget=1)
    #
    # print("*** Example 4: Generated lobster-turned-chordal graph ***")
    # G4 = nx.complete_to_chordal_graph(nx.random_lobster(n=50, p1=0.3, p2=0.2))
    # print(G4[0])
    # run_example(G4[0], root=0, min_save=10, budget=1)
