from collections import defaultdict
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt


def find_swappable_pairs(colored_lists):
    AllLists = {}
    ListsByElem = {}
    colors = colored_lists.keys()
    for c in colors:
        AllLists[c] = set()
        ListsByElem[c] = defaultdict(set)
        for L in colored_lists[c]:
            fs = frozenset(L)
            AllLists[c].add(fs)
            for i in fs:
                ListsByElem[c][i].add(fs)

    signature = {}
    all_elements = set()
    for c in colors:
        all_elements |= set(ListsByElem[c].keys())
    for i in all_elements:
        signature[i] = tuple(len(ListsByElem[c].get(i, ())) for c in colors)

    sig_groups = defaultdict(list)
    for i, sig in signature.items():
        sig_groups[sig].append(i)

    swappable = []
    for element in sig_groups.values():
        for x, y in combinations(element, 2):
            valid = True
            for c in colors:
                Lx = ListsByElem[c].get(x, set())
                Ly = ListsByElem[c].get(y, set())
                both = Lx & Ly
                A = Lx - both
                B = Ly - both
                if len(A) != len(B):
                    valid = False
                    break
                for L in A:
                    swapped = frozenset((L - {x}) | {y})
                    if swapped not in AllLists[c]:
                        valid = False
                        break
                if not valid:
                    break
                for L in B:
                    swapped = frozenset((L - {y}) | {x})
                    if swapped not in AllLists[c]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                swappable.append((x, y))
    return swappable


def plot_group_graph(group_lists, group_name="group"):
    """
    group_lists: list of lists (e.g. from colored_lists['red'])
    Displays a bipartite graph of elements vs list-nodes
    """
    G = nx.Graph()
    element_nodes = set()
    list_nodes = []

    for idx, L in enumerate(group_lists):
        list_id = f"L{idx}"
        list_nodes.append(list_id)
        G.add_node(list_id, bipartite=0, label="list")
        for elem in L:
            G.add_node(elem, bipartite=1, label="element")
            G.add_edge(list_id, elem)
            element_nodes.add(elem)

    # Layout: bipartite
    pos = {}
    pos.update((n, (1, i)) for i, n in enumerate(sorted(element_nodes)))   # top row
    pos.update((n, (0, i)) for i, n in enumerate(list_nodes))              # bottom row

    plt.figure(figsize=(8, 5))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=800,
        node_color=["lightblue" if n in element_nodes else "lightgreen" for n in G.nodes],
        edge_color="gray"
    )
    plt.title(f"Incidence Graph for Group '{group_name}'")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Example usage ---
    data2 = {
        'red': [[1, 2], [1, 4], [2, 6]],
        'blue': [[5, 3]],
        'green': [[6, 7]]
    }

    print("Swappable pairs:", find_swappable_pairs(data2))
    plot_group_graph(data2['red'], group_name="red")
