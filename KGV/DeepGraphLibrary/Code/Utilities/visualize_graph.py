# Purpose: Visualize graph by setting the nodes and edges.
# Execution: python3 visualize_graph.py

import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()
G.graph["Name"] = "subset_genomics_graph"
plt.figure(figsize=(12, 7))

G.add_nodes_from([
    (0, {"id": 0, "originID":"828371", "variantID": "rs3752420", "accessionID":"SRR12712806"}), 
    (1, {"id": 1, "originID":"1854537", "variantID": "", "accessionID":"SRR12712806"}), 
    (2, {"id": 2, "originID":"2192637", "variantID": "", "accessionID":"SRR12712806"}),
    (3, {"id": 3, "originID":"1465744", "variantID": "", "accessionID":"SRR12712806"}),
    (4, {"id": 4, "originID":"1221206", "variantID": "", "accessionID":"SRR12712806"}),

    (5, {"id": 5, "originID":"834490", "variantID": "rs3752420", "accessionID":"SRR12712827"}),
    (6, {"id": 6, "originID":"1718800", "variantID": "", "accessionID":"SRR12712827"}),
    (7, {"id": 7, "originID":"1338352", "variantID": "", "accessionID":"SRR12712827"}),
    (8, {"id": 8, "originID":"846098", "variantID": "", "accessionID":"SRR12712827"}),
    (9, {"id": 9, "originID":"2189972", "variantID": "", "accessionID":"SRR12712827"}),
])

G.add_edges_from([
    (0, 1, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (1, 0, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (1, 2, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (2, 1, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (2, 3, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (3, 2, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (3, 4, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (4, 3, {"weight": 1, "variantID": " ", "color": "#0066CC"}),

    (0, 5, {"weight": 1, "variantID": "rs3752420", "color": "#FF9933"}),
    (5, 0, {"weight": 1, "variantID": "rs3752420", "color": "#FF9933"}),

    (5, 6, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (6, 7, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (7, 8, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (8, 9, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (6, 5, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (7, 6, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (8, 7, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
    (9, 8, {"weight": 1, "variantID": " ", "color": "#0066CC"}),
])

for n, d in G.nodes(data=True):
    print(n)
    print(d)

node_labels = {n: (d["id"], d["accessionID"]) for n, d in G.nodes(data=True)}
edge_labels = {(u,v): (d["weight"]) for u,v,d in G.edges(data=True)}
colors = [G[u][v]["color"] for u,v,d in G.edges(data=True)]

pos = nx.circular_layout(G)
nx.draw(G, pos=pos, with_labels=False, node_color="#99CCFF", node_size=1500, edge_color=colors, font_size=25, width=2, arrows=True, arrowsize=20)
nx.draw_networkx_labels(G, pos=pos, labels=node_labels)
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

plt.margins(0.2)
plt.savefig("graph_4.png")