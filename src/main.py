import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
import matplotlib.colors as mcolors
import random

st.set_page_config(page_title="Route Planner", layout="wide")
st.title("🗺️ Interactive Route Planner with Avoid Zones")

# Sidebar controls
st.sidebar.header("Route Settings")
num_paths = st.sidebar.slider("Number of alternate paths", 1, 5, 3)
penalty_factor = st.sidebar.slider("Path diversity factor", 1.0, 3.0, 1.5, 0.1)
avoid_penalty = st.sidebar.slider("Avoid zone penalty (per meter)", 0.0, 10.0, 2.0, 0.1)
show_blockages = st.sidebar.checkbox("Show blockages", True)
show_avoid_zones = st.sidebar.checkbox("Show avoid zones", True)

@st.cache_data
def load_base_graph():
    """Load graph and remove blocked edges (cached, runs once)"""
    G = ox.graph_from_xml("data/map")
    
    # Define hard blockages (completely blocked)
    blockages = [
        Polygon([(-86.80680, 36.14580), (-86.80650, 36.14580), (-86.80650, 36.14610), (-86.80680, 36.14610)]),
        Polygon([(-86.80550, 36.14720), (-86.80520, 36.14720), (-86.80520, 36.14750), (-86.80550, 36.14750)]),
        Polygon([(-86.80480, 36.14650), (-86.80460, 36.14650), (-86.80460, 36.14690), (-86.80480, 36.14690)]),
        Polygon([(-86.80300, 36.14800), (-86.80270, 36.14800), (-86.80270, 36.14830), (-86.80300, 36.14830)]),
        Polygon([(-86.80420, 36.14540), (-86.80380, 36.14540), (-86.80380, 36.14560), (-86.80420, 36.14560)]),
        Polygon([(-86.80700, 36.14650), (-86.80670, 36.14650), (-86.80670, 36.14680), (-86.80700, 36.14680)]),
        Polygon([(-86.80600, 36.14480), (-86.80520, 36.14480), (-86.80520, 36.14500), (-86.80600, 36.14500)]),
        Polygon([(-86.80850, 36.14400), (-86.80820, 36.14400), (-86.80820, 36.14440), (-86.80850, 36.14440)])
    ]
    
    # Remove completely blocked edges
    edges_to_remove = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'geometry' in data:
            edge_geom = data['geometry']
        else:
            u_point = (G.nodes[u]['x'], G.nodes[u]['y'])
            v_point = (G.nodes[v]['x'], G.nodes[v]['y'])
            edge_geom = LineString([u_point, v_point])
        
        # Check for complete blockages
        for blockage in blockages:
            if edge_geom.intersects(blockage):
                edges_to_remove.append((u, v, key))
                break
    
    G.remove_edges_from(edges_to_remove)
    
    # Store original lengths for penalty calculations
    for u, v, key, data in G.edges(keys=True, data=True):
        G[u][v][key]['original_length'] = data.get('length', 100)
    
    return G, blockages, len(edges_to_remove)

@st.cache_data
def get_avoid_zones():
    """Define avoid zones (cached)"""
    return [
        # Large residential area to avoid (northwest)
        Polygon([
            (-86.81000, 36.14300),
            (-86.80600, 36.14300),
            (-86.80600, 36.14600),
            (-86.81000, 36.14600)
        ]),
        # Busy commercial district (central area)
        Polygon([
            (-86.80650, 36.14500),
            (-86.80350, 36.14500),
            (-86.80350, 36.14750),
            (-86.80650, 36.14750)
        ]),
        # School zone (northeast)
        Polygon([
            (-86.80400, 36.14750),
            (-86.80150, 36.14750),
            (-86.80150, 36.14950),
            (-86.80400, 36.14950)
        ]),
        # Construction zone (south)
        Polygon([
            (-86.80800, 36.14200),
            (-86.80400, 36.14200),
            (-86.80400, 36.14400),
            (-86.80800, 36.14400)
        ]),
        # High traffic corridor (east-west through middle)
        Polygon([
            (-86.80900, 36.14600),
            (-86.80200, 36.14600),
            (-86.80200, 36.14700),
            (-86.80900, 36.14700)
        ])
    ]

@st.cache_data
def calculate_avoid_zone_intersections(_G_edges, _avoid_zones):
    """Pre-calculate which edges intersect avoid zones and by how much (cached)"""
    edge_penalties = {}
    
    for u, v, key, data in _G_edges:
        if 'geometry' in data:
            edge_geom = data['geometry']
        else:
            u_point = (data['u_x'], data['u_y'])
            v_point = (data['v_x'], data['v_y'])
            edge_geom = LineString([u_point, v_point])
        
        total_penalty_length = 0
        for avoid_zone in _avoid_zones:
            if edge_geom.intersects(avoid_zone):
                try:
                    intersection = edge_geom.intersection(avoid_zone)
                    if hasattr(intersection, 'length'):
                        # Convert to meters (approximate)
                        intersection_length_meters = intersection.length * 111000
                        total_penalty_length += intersection_length_meters
                except:
                    # Fallback: use partial edge length if intersection calculation fails
                    edge_length_meters = data.get('original_length', 100)
                    total_penalty_length += edge_length_meters * 0.5
        
        if total_penalty_length > 0:
            edge_penalties[(u, v, key)] = total_penalty_length
    
    return edge_penalties

def apply_avoid_penalties(G, edge_penalties, avoid_penalty_per_meter):
    """Apply current avoid zone penalties to graph (fast operation)"""
    G_copy = G.copy()
    penalties_applied = 0
    
    for (u, v, key), penalty_length in edge_penalties.items():
        if G_copy.has_edge(u, v):
            original_length = G_copy[u][v][key]['original_length']
            penalty = penalty_length * avoid_penalty_per_meter
            G_copy[u][v][key]['length'] = original_length + penalty
            penalties_applied += 1
    
    return G_copy, penalties_applied

def find_alternate_paths(G, source, target, num_paths=3, penalty_factor=1.5):
    """Find alternate paths by penalizing previously used edges"""
    paths = []
    G_copy = G.copy()
    
    for i in range(num_paths):
        try:
            path = nx.shortest_path(G_copy, source, target, weight='length')
            paths.append(path)
            
            # Penalize edges in this path for next iteration
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                for key in G_copy[u][v]:
                    G_copy[u][v][key]['length'] *= penalty_factor
                    
        except nx.NetworkXNoPath:
            break
    
    return paths

def generate_colors(n):
    colors = []
    for _ in range(n):
        h = random.random()
        s = random.uniform(0.6, 0.9)
        v = random.uniform(0.7, 0.9)
        rgb = mcolors.hsv_to_rgb([h, s, v])
        hex_color = mcolors.rgb2hex(rgb)
        colors.append(hex_color)
    return colors

# Initialize session state
if 'start_point' not in st.session_state:
    st.session_state.start_point = [-86.80867561015874, 36.14420758854631]  # Stadium
if 'end_point' not in st.session_state:
    st.session_state.end_point = [-86.80282365449781, 36.14819247137839]   # Kirkland

# Load base graph (only runs once due to caching)
try:
    with st.spinner("Loading graph (one-time setup)..."):
        base_G, blockages, removed_edges = load_base_graph()
        avoid_zones = get_avoid_zones()
        
        # Pre-calculate avoid zone intersections (also cached)
        G_edges_data = []
        for u, v, key, data in base_G.edges(keys=True, data=True):
            edge_data = data.copy()
            edge_data['u_x'] = base_G.nodes[u]['x']
            edge_data['u_y'] = base_G.nodes[u]['y'] 
            edge_data['v_x'] = base_G.nodes[v]['x']
            edge_data['v_y'] = base_G.nodes[v]['y']
            G_edges_data.append((u, v, key, edge_data))
        
        edge_penalties = calculate_avoid_zone_intersections(G_edges_data, avoid_zones)
        
    st.sidebar.success(f"Graph loaded! 🚫 {removed_edges} blocked edges")
    st.sidebar.info(f"⚠️ {len(edge_penalties)} edges in avoid zones")
    
except Exception as e:
    st.error(f"Error loading graph: {str(e)}")
    st.stop()

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Create folium map
    m = folium.Map(
        location=[36.144, -86.805],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Add markers for start and end points
    folium.Marker(
        [st.session_state.start_point[1], st.session_state.start_point[0]],
        popup="Start Point",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        [st.session_state.end_point[1], st.session_state.end_point[0]],
        popup="End Point",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # Add blockages if enabled
    if show_blockages:
        for i, blockage in enumerate(blockages):
            coords = [(lat, lon) for lon, lat in blockage.exterior.coords]
            folium.Polygon(
                coords,
                color='red',
                weight=2,
                fillColor='red',
                fillOpacity=0.5,
                popup=f'🚫 Blockage {i+1} (Complete block)'
            ).add_to(m)
    
    # Add avoid zones if enabled
    if show_avoid_zones:
        zone_names = ['Residential Area', 'Commercial District', 'School Zone', 'Construction Zone', 'High Traffic Corridor']
        for i, avoid_zone in enumerate(avoid_zones):
            coords = [(lat, lon) for lon, lat in avoid_zone.exterior.coords]
            folium.Polygon(
                coords,
                color='orange',
                weight=2,
                fillColor='yellow',
                fillOpacity=0.2,
                popup=f'⚠️ {zone_names[i]} (Penalty: {avoid_penalty}/m)'
            ).add_to(m)

    # Calculate routes button
    if st.sidebar.button("Calculate Routes", type="primary"):
        with st.spinner("Calculating routes..."):
            try:
                # Apply current avoid penalties (fast operation)
                G_with_penalties, penalties_applied = apply_avoid_penalties(base_G, edge_penalties, avoid_penalty)
                
                # Convert points to nodes
                start_point = Point(st.session_state.start_point[0], st.session_state.start_point[1])
                end_point = Point(st.session_state.end_point[0], st.session_state.end_point[1])
                
                start_node = ox.distance.nearest_nodes(G_with_penalties, X=start_point.x, Y=start_point.y)
                end_node = ox.distance.nearest_nodes(G_with_penalties, X=end_point.x, Y=end_point.y)
                
                # Calculate routes
                routes = find_alternate_paths(G_with_penalties, start_node, end_node, num_paths, penalty_factor)
                
                if routes:
                    # Generate colors and add routes to map
                    colors = generate_colors(len(routes))
                    
                    for i, route in enumerate(routes):
                        coords = [(G_with_penalties.nodes[node]['y'], G_with_penalties.nodes[node]['x']) for node in route]
                        folium.PolyLine(
                            coords,
                            color=colors[i],
                            weight=4,
                            opacity=0.8,
                            popup=f'🛣️ Route {i+1} ({len(route)} nodes)'
                        ).add_to(m)
                    
                    st.session_state.routes = routes
                    st.session_state.route_colors = colors
                    st.session_state.current_graph = G_with_penalties
                    st.success(f"Found {len(routes)} alternate paths! (Applied {penalties_applied} penalties)")
                else:
                    st.error("No routes found!")
                    
            except nx.NetworkXNoPath:
                st.error("No path available - points may be disconnected by blockages")
            except Exception as e:
                st.error(f"Error calculating routes: {str(e)}")
    
    # Display existing routes if they exist
    if 'routes' in st.session_state and 'current_graph' in st.session_state:
        for i, route in enumerate(st.session_state.routes):
            coords = [(st.session_state.current_graph.nodes[node]['y'], 
                      st.session_state.current_graph.nodes[node]['x']) for node in route]
            folium.PolyLine(
                coords,
                color=st.session_state.route_colors[i],
                weight=4,
                opacity=0.8,
                popup=f'🛣️ Route {i+1} ({len(route)} nodes)'
            ).add_to(m)
    
    # Display the interactive map
    map_data = st_folium(m, width=700, height=500, returned_objects=["last_clicked"])
    
    # Handle map clicks to set new start/end points
    if map_data['last_clicked']:
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lng = map_data['last_clicked']['lng']
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Set as Start Point"):
                st.session_state.start_point = [clicked_lng, clicked_lat]
                if 'routes' in st.session_state:
                    del st.session_state.routes
                st.rerun()
        with col_b:
            if st.button("Set as End Point"):
                st.session_state.end_point = [clicked_lng, clicked_lat]
                if 'routes' in st.session_state:
                    del st.session_state.routes
                st.rerun()

with col2:
    st.subheader("Route Information")
    
    # Display current points
    st.write("**Start Point:**")
    st.write(f"Lat: {st.session_state.start_point[1]:.6f}")
    st.write(f"Lng: {st.session_state.start_point[0]:.6f}")
    
    st.write("**End Point:**")
    st.write(f"Lat: {st.session_state.end_point[1]:.6f}")
    st.write(f"Lng: {st.session_state.end_point[0]:.6f}")
    
    # Route statistics
    if 'routes' in st.session_state and 'current_graph' in st.session_state:
        st.write("**Route Statistics:**")
        G_current = st.session_state.current_graph
        for i, route in enumerate(st.session_state.routes):
            # Calculate total route length
            total_length = 0
            for j in range(len(route) - 1):
                u, v = route[j], route[j+1]
                edge_data = G_current[u][v][0]  # Take first edge if multiple
                total_length += edge_data.get('length', 0)
            
            st.write(f"🔸 Route {i+1}: {len(route)} nodes, {total_length:.0f}m")
    
    # Zone legend
    st.write("**Zone Types:**")
    st.write("🚫 **Blockages** - Complete blocks")
    st.write("⚠️ **Avoid Zones** - Penalty areas")
    st.write(f"Current penalty: {avoid_penalty}/m")

# Instructions
st.sidebar.markdown("""
### Performance Notes:
- ✅ Graph loads once (cached)
- ✅ Penalty slider is instant
- ✅ Only route calculation runs when needed

### How to use:
1. 🖱️ Click map to set start/end points
2. ⚙️ Adjust avoid zone penalty (instant)
3. ▶️ Click "Calculate Routes"
4. 🔍 Routes avoid penalty zones efficiently
""")

st.sidebar.markdown("""
### Zone Types:
- 🏠 **Residential** - Quiet neighborhoods
- 🏢 **Commercial** - Busy shopping areas  
- 🏫 **School** - Safety zones
- 🚧 **Construction** - Work zones
- 🚦 **High Traffic** - Congested roads
""")
