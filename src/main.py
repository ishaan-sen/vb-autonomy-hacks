import osmnx as ox

# Load from a local file
graph = ox.graph_from_xml("data/map")

# Convert to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(graph)

# Preview
print(edges.head())
ox.plot_graph(graph)
