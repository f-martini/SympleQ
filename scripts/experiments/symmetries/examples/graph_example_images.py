# Create illustrative images for the example:
# - Input graph (edge-colored complete graph from S)
# - WL refinement (base color classes)
# - Permutation search (partial mapping and final mapping)
#
# This is a didactic example using V = {1,2,3,4} and a simple S that's
# invariant under any permutation (all-ones off-diagonal over GF(2)).

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# -------------------- Example data --------------------
labels = [1,2,3,4]
n = len(labels)

# Symplectic product matrix over GF(2): all ones off-diagonal, zero diag.
S = np.ones((n,n), dtype=int) - np.eye(n, dtype=int)

# WL-1 base classes seeded by row histograms (here all equal -> one class)
def wl1_base_colors(S_mod, p=2):
    n = S_mod.shape[0]
    hist = np.array([np.bincount(S_mod[i], minlength=p) for i in range(n)])
    # palette based on (row-histogram) only for this demo
    uniq, inv = np.unique(hist, axis=0, return_inverse=True)
    return inv  # color id per node

base_colors = wl1_base_colors(S % 2, p=2)  # all zeros here
# Build class listing
classes = {}
for idx, c in enumerate(base_colors):
    classes.setdefault(int(c), []).append(labels[idx])

# A permutation from the LaTeX example (cycle notation (1 3 2 4))
# Map in 1-based indices: 1->3, 2->4, 3->2, 4->1
perm_map = {1:3, 2:4, 3:2, 4:1}
perm_partial = {1:3, 4:1}  # the partial bijection from the example

# -------------------- 1) Input graph --------------------
G = nx.complete_graph(labels)  # undirected complete graph

pos = nx.circular_layout(G, scale=2)  # stable positions

plt.figure(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_size=1200, font_size=14)
# annotate edge colors (S_ij values) on one triangle to avoid clutter
edge_labels = {}
for i in labels:
    for j in labels:
        if i < j:
            edge_labels[(i,j)] = f"{S[labels.index(i), labels.index(j)]}"
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
plt.title("Input graph (edge-coloured by $S_{ij}$ over GF(2))")
plt.axis('off')
plt.tight_layout()

# -------------------- 2) WL refinement (base partition) --------------------
plt.figure(figsize=(6, 6))
# color nodes by base_colors (matplotlib default colormap; didactic only)
node_color = base_colors  # ints; matplotlib maps to default colors
nx.draw(G, pos, with_labels=True, node_size=1200, font_size=14, node_color=node_color, cmap=None)
# Annotate each node with its class id
for i in labels:
    xy = pos[i]
    plt.text(xy[0], xy[1]-0.15, f"class {base_colors[labels.index(i)]}", ha='center', va='top', fontsize=10)
plt.title("WL-1 base partition (all nodes coarsely equivalent in this toy $S$)")
plt.axis('off')
plt.tight_layout()
# plt.savefig(fig2_path, dpi=180)

# -------------------- 3) Permutation search: partial and final --------------------
# We'll draw a bipartite "mapping" diagram: domain (left) to codomain (right)

def draw_mapping(mapping, title, path):
    left_y = np.linspace(1, 0, n)
    right_y = np.linspace(1, 0, n)
    left_pos = {labels[i]: (-1.0, left_y[i]) for i in range(n)}
    right_pos = {labels[i]: (1.0, right_y[i]) for i in range(n)}

    plt.figure(figsize=(7, 5))
    # left and right node sets
    for i in labels:
        x,y = left_pos[i]
        plt.scatter([x],[y], s=900)
        plt.text(x, y, f"{i}", ha='center', va='center', fontsize=14, color='white')
    for i in labels:
        x,y = right_pos[i]
        plt.scatter([x],[y], s=900)
        plt.text(x, y, f"{i}", ha='center', va='center', fontsize=14, color='white')

    # arrows for mapped pairs
    for i, j in mapping.items():
        x0,y0 = left_pos[i]
        x1,y1 = right_pos[j]
        plt.annotate("",
                     xy=(x1-0.05,y1),
                     xytext=(x0+0.05,y0),
                     arrowprops=dict(arrowstyle="->", lw=2))
        # inline label
        xm, ym = (x0+x1)/2, (y0+y1)/2
        plt.text(xm, ym+0.03, f"{i}\u2192{j}", ha='center', va='bottom', fontsize=10)

    # cosmetics
    plt.text(-1.0, 1.05, "domain", ha='center', va='bottom', fontsize=12)
    plt.text(1.0, 1.05, "codomain", ha='center', va='bottom', fontsize=12)
    plt.title(title)
    plt.axis('off')
    plt.xlim(-1.3, 1.3); plt.ylim(-0.1, 1.1)
    plt.tight_layout()

# Partial mapping (1->3, 4->1) as in the LaTeX example
fig3_path = "/mnt/data/mapping_partial.png"
draw_mapping(perm_partial, "Permutation search: partial bijection $\\phi=\\{1\\mapsto 3,\\ 4\\mapsto 1\\}$", fig3_path)

# Final mapping (1->3, 2->4, 3->2, 4->1)
fig4_path = "/mnt/data/mapping_final.png"
draw_mapping(perm_map, "Permutation search: completed permutation $\\Pi=(1\\ 3\\ 2\\ 4)$", fig4_path)

# (fig1_path, fig2_path, fig3_path, fig4_path)

plt.show()

