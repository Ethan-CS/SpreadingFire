import networkx as nx
import numpy as np
import pandas as pd

from dp_sfire_helpers import find_problems, is_chordal, create_clique_tree


# INFO this is my first attempt, iterative rather than recursive so second attempt is in 'from_scratch`

# TODO budget implementation not working, check logic (works fine with budget=1)
def s_fire(graph, root, min_save, budget=1, problem='OPT', p=False):
    if not is_chordal(graph):
        raise Exception('Graph is not chordal!')

    clique_tree, cliques, root_clique = create_clique_tree(graph, root, root)
    if p:
        print("clique tree\n", clique_tree, "\ncliques", cliques)

    dp_optimal = {}
    dp_decision = pd.DataFrame(columns=['clique', 'fire state', 'time', 'value'])

    burning = {root}

    # Process cliques in reverse order (bottom-up)
    for clique_index in range(len(cliques) - 1, -1, -1):
        if p:
            print('processing clique', cliques[clique_index])
        child_indices = np.where(clique_tree[clique_index] == 1)[0]
        if len(child_indices) == 0:
            if p:
                print('  ! this is a leaf clique, move onto next')
            continue  # Skip leaf cliques as they're already processed
        if p:
            print("try and defend children:", [cliques[child_index] for child_index in child_indices])
        for fire_state in [False, True]:
            for time_left in range(min_save + 1):
                clique_considering = set(cliques[clique_index])
                num_can_defend = len([v for v in clique_considering if v not in burning]) if not fire_state else min(
                    budget, len(clique_tree[clique_index]))

                running_max_defended = 0
                upper_bound = min(time_left * budget, num_can_defend) if not fire_state else min(budget, len(
                    clique_tree[clique_index]))
                for k in range(upper_bound + 1):
                    if 'OPT' in problem:
                        child_fire_state = False if not fire_state and k > 0 else 1
                        defended_here = k
                        defended_children = sum(
                            dp_optimal.get(
                                (child_index, child_fire_state, max(0, time_left - (k + budget - 1) // budget)), 0)
                            for child_index in child_indices)
                        total_defended = defended_here + defended_children
                        running_max_defended = max(running_max_defended, total_defended)
                if 'OPT' in problem:
                    dp_optimal[(clique_index, fire_state, time_left)] = running_max_defended
                if p:
                    print(
                        f' setting dp[{cliques[clique_index]}][{fire_state}][{time_left}] value to: {running_max_defended >= min_save}')
                dp_decision.loc[len(dp_decision)] = {'clique': cliques[clique_index], 'fire state': fire_state,
                                                     'time': min_save + 1 - time_left,
                                                     'value': running_max_defended >= min_save}
    if p:
        print("\nDP Table:")
        print(dp_decision)
    if 'OPT' in problem:
        defended_vertices, burning_by_time, saved_vertices = reconstruct_defended_vertices(graph, cliques, min_save,
                                                                                           root,
                                                                                           dp_optimal,
                                                                                           budget, p)
        success = len(saved_vertices) >= min_save
        find_problems(success, defended_vertices, saved_vertices, burning_by_time, min_save)
        if p:
            print(f"Success: {success}, saved {len(saved_vertices)} vertices and needed to save at least {min_save}")
        return success, defended_vertices, saved_vertices, burning_by_time
    else:
        index_of_root_clique = -1
        for i, K in enumerate(cliques):
            if K == root_clique:
                index_of_root_clique = i
                break
        print(dp_decision)
        success = dp_decision[(dp_decision['clique'] == root_clique) & (dp_decision['fire state'] == True) & (
                    dp_decision['time'] == min_save), 'value']
        return success


def s_fire_decision(graph, root, min_save, budget=1, p=False):
    if not is_chordal(graph):
        raise Exception('Graph is not chordal!')

    clique_tree, cliques, root_clique = create_clique_tree(graph, root, root)
    print("clique tree\n", clique_tree, "\ncliques", cliques)

    dp_optimal = {}
    dp_decision = pd.DataFrame(columns=['clique', 'fire state', 'time', 'value'])
    burning = {root}
    # Process cliques in reverse order (bottom-up)
    for clique_index in range(len(cliques) - 1, -1, -1):
        child_indices = np.where(clique_tree[clique_index] == 1)[0]
        if len(child_indices) == 0:  # ! this is a leaf clique, move onto next
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
                    upper_bound = min(time_left * budget, len(can_defend)) + 1
                else:
                    upper_bound = min(budget, len(clique_tree[clique_index])) + 1

                running_max_defended = 0
                for k in range(upper_bound):
                    defended_here = k
                    defended_children = sum(
                        dp_optimal.get((child_index, child_fire_state, max(0, time_left - (k + budget - 1) // budget)), 0)
                        for child_index in child_indices)
                    total_defended = defended_here + defended_children
                    running_max_defended = max(running_max_defended, total_defended)
                dp_optimal[(clique_index, fire_state, time_left)] = running_max_defended
                dp_decision.loc[len(dp_decision)] = {'clique': clique_index, 'fire state': fire_state,
                                                     'time': min_save + 1 - time_left,
                                                     'value': running_max_defended >= min_save}
    defended_vertices, burning_by_time, saved_vertices = reconstruct_defended_vertices(graph, cliques, min_save, root,
                                                                                       dp_optimal,
                                                                                       budget, p)
    success = len(saved_vertices) >= min_save
    find_problems(success, defended_vertices, saved_vertices, burning_by_time, min_save)
    print(f"Success: {success}, saved {len(saved_vertices)} vertices and needed to save at least {min_save}")
    print('HERE')

    can_we = False
    for i in range(len(dp_decision)):
        if dp_decision.iloc[i]['clique'] == cliques.index(root_clique) and dp_decision.iloc[i]['fire state'] and dp_decision.iloc[i]['value']:
            print(dp_decision.iloc[i])
            can_we = True
    return can_we


def reconstruct_defended_vertices(graph, cliques, min_save, root, dp_optimisation, budget=1, p=True):
    defended_by_time = {}
    all_defended = set()
    burning = {root}
    burning_by_time = {}
    time = 0

    while time < min_save:
        defended_by_time[time] = {'direct': set(), 'spread': set()}
        burning_by_time[time] = set()
        if p:
            print(f"Time {time}: Defended by time: {defended_by_time}, Burning: {burning}")
        # Choose vertices to defend
        defensible = [v for v in graph.nodes() if v not in burning and v not in all_defended]
        for _ in range(budget):
            if defensible:
                best_vertex = max(defensible, key=lambda v: max(
                    dp_optimisation.get((ci, 0, min_save - time), 0) - dp_optimisation.get((ci, 1, min_save - time), 0)
                    for ci, c in
                    enumerate(cliques) if v in c))
                defended_by_time[time]['direct'].add(best_vertex)
                all_defended.add(best_vertex)
                defensible.remove(best_vertex)

        new_burning, new_defended = spread_and_update(time, all_defended, burning, defended_by_time, graph,
                                                      burning_by_time, p)

        time += 1

        if not new_burning and not new_defended:
            break  # Stop if no changes occurred

    return defended_by_time, burning_by_time, list(set(graph.nodes()) - burning)


def spread_and_update(time, all_defended, burning, defended, graph, burning_at_time, p=False):
    # Fire spread
    new_burning = set()
    for v in burning:
        for n in graph.neighbors(v):
            if n not in all_defended and n not in burning:
                new_burning.add(n)
    if p:
        print('want to burn', new_burning, 'next')
    burning.update(new_burning)
    burning_at_time[time].update(new_burning)

    # Defence spread
    new_defended = set()
    for d in all_defended:
        for neighbor in graph.neighbors(d):
            if neighbor not in burning and neighbor not in all_defended:
                new_defended.add(neighbor)
    if p:
        print('want to spread defence to', new_defended, 'next')
    defended[time]['spread'].update(new_defended)
    all_defended.update(new_defended)

    return new_burning, new_defended


# Example usage
if __name__ == "__main__":
    def run_example(G, root, min_save, budget):
        # success, defended, saved, burning_by_time = s_fire(G, root, min_save, budget)
        success = s_fire_decision(G, root, min_save, budget)
        # find_problems(success, defended, saved, burning_by_time, min_save)
        # burning_number = len(list(set(G.nodes - set(saved))))
        # print('Success?', success)
        # print(' *', len(saved), 'saved by defending', f'{len(union_of_defended_dict(defended))}, target min. was',
        #       min_save)
        # print(' * There are', burning_number, 'burning, saved + burning =', len(saved) + burning_number, 'should be',
        #       len(G.nodes))
        # print(" * Defended vertices:")
        # print(defended)
        # print(" * Saved vertices:")
        # print(saved)
        # print(f" * Burning vertices:")
        # print(burning_by_time)
        # print(set(G.nodes()) - set(saved))
        # print()


    print(" *** Example 1: Small chordal graph ***")
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
    run_example(G1, root=0, min_save=3, budget=1)

    print(" *** Example 2: Larger chordal graph ***")
    G2 = nx.Graph()
    G2.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0), (0, 2),  # Chordal cycle
        (3, 4), (4, 5), (5, 6), (6, 4)  # Rest of the graph
    ])
    run_example(G2, root=0, min_save=5, budget=1)
    #
    # print(" *** Example 3: Complete graph ***")
    # run_example(nx.complete_graph(6), root=0, min_save=4, budget=1)
    #
    # print("*** Example 4: Generated lobster-turned-chordal graph ***")
    # G4 = nx.complete_to_chordal_graph(nx.random_lobster(n=50, p1=0.3, p2=0.2))
    # print(G4[0])
    # run_example(G4[0], root=0, min_save=10, budget=1)
