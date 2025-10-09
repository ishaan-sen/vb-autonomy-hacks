import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
import matplotlib.colors as mcolors
import pandas as pd
import random

st.set_page_config(page_title="pathPlanner", layout="wide")
st.title("AutoGators Convoy Routing UI")

# Sidebar controls
st.sidebar.header("Route Settings")
num_paths = st.sidebar.slider("Number of alternate paths", 1, 5, 3)
penalty_factor = st.sidebar.slider("Path diversity factor", 1.0, 3.0, 1.5, 0.1)
avoid_penalty = st.sidebar.slider("Hostile area penalty (per meter)", 0.0, 10.0, 2.0, 0.1)
show_blockages = st.sidebar.checkbox("Show threat zones", True)
show_avoid_zones = st.sidebar.checkbox("Show hostile areas", True)

@st.cache_data
@st.cache_data
def load_threats_from_csv():
    """Load threat/hazard polygons from S3 or local CSV file (cached). Falls back silently to defaults."""
    import boto3
    import io

    bucket_name = os.environ.get("THREAT_S3_BUCKET", "autogators-data")  # optional env var
    object_key = os.environ.get("THREAT_S3_KEY", "data/threats.csv")

    def parse_threats_from_df(df):
        threats = []
        for _, row in df.iterrows():
            try:
                polygon = Polygon([
                    (row['lon_min'], row['lat_min']),
                    (row['lon_max'], row['lat_min']),
                    (row['lon_max'], row['lat_max']),
                    (row['lon_min'], row['lat_max']),
                    (row['lon_min'], row['lat_min'])
                ])
                threats.append({
                    'polygon': polygon,
                    'stem': row.get('stem', f"THREAT_{_}"),
                    'class': row.get('class', 'unknown'),
                    'class_id': int(row.get('class_id', _))
                })
            except Exception:
                continue
        return threats

    # --- Try to load from S3 ---
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket_name, Key=object_key)
        csv_data = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        return parse_threats_from_df(df)
    except Exception:
        pass  # silently fail and try local CSV

    # --- Try to load from local file ---
    try:
        df = pd.read_csv("data/threats.csv")
        return parse_threats_from_df(df)
    except Exception:
        pass  # silently fail and use defaults

    # --- Final fallback: hardcoded threats ---
    return [
        {'polygon': Polygon([(-86.80680, 36.14580), (-86.80650, 36.14580), (-86.80650, 36.14610), (-86.80680, 36.14610)]), 'stem': 'IED_1', 'class': 'explosive', 'class_id': 1},
        {'polygon': Polygon([(-86.80550, 36.14720), (-86.80520, 36.14720), (-86.80520, 36.14750), (-86.80550, 36.14750)]), 'stem': 'MINE_1', 'class': 'landmine', 'class_id': 2},
        {'polygon': Polygon([(-86.80480, 36.14650), (-86.80460, 36.14650), (-86.80460, 36.14690), (-86.80480, 36.14690)]), 'stem': 'UXO_1', 'class': 'ordnance', 'class_id': 3},
        {'polygon': Polygon([(-86.80300, 36.14800), (-86.80270, 36.14800), (-86.80270, 36.14830), (-86.80300, 36.14830)]), 'stem': 'IED_2', 'class': 'explosive', 'class_id': 4},
        {'polygon': Polygon([(-86.80420, 36.14540), (-86.80380, 36.14540), (-86.80380, 36.14560), (-86.80420, 36.14560)]), 'stem': 'MINE_2', 'class': 'landmine', 'class_id': 5},
        {'polygon': Polygon([(-86.80700, 36.14650), (-86.80670, 36.14650), (-86.80670, 36.14680), (-86.80700, 36.14680)]), 'stem': 'UXO_2', 'class': 'ordnance', 'class_id': 6},
        {'polygon': Polygon([(-86.80600, 36.14480), (-86.80520, 36.14480), (-86.80520, 36.14500), (-86.80600, 36.14500)]), 'stem': 'ROADBLOCK_1', 'class': 'obstacle', 'class_id': 7},
        {'polygon': Polygon([(-86.80850, 36.14400), (-86.80820, 36.14400), (-86.80820, 36.14440), (-86.80850, 36.14440)]), 'stem': 'IED_3', 'class': 'explosive', 'class_id': 8}
    ]


@st.cache_data
def load_base_graph():
    """Load graph and remove edges blocked by threats/hazards (cached, runs once)"""
    G = ox.graph_from_xml("data/map")
    # Load threats from CSV
    threat_data = load_threats_from_csv()
    threat_polygons = [item['polygon'] for item in threat_data]
    # Remove edges completely blocked by threats (landmines, IEDs, etc.)
    edges_to_remove = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'geometry' in data:
            edge_geom = data['geometry']
        else:
            u_point = (G.nodes[u]['x'], G.nodes[u]['y'])
            v_point = (G.nodes[v]['x'], G.nodes[v]['y'])
            edge_geom = LineString([u_point, v_point])
        # Check for complete blockages from threats
        for threat_polygon in threat_polygons:
            if edge_geom.intersects(threat_polygon):
                edges_to_remove.append((u, v, key))
                break
    G.remove_edges_from(edges_to_remove)
    # Store original lengths for penalty calculations
    for u, v, key, data in G.edges(keys=True, data=True):
        G[u][v][key]['original_length'] = data.get('length', 100)
    return G, threat_data, len(edges_to_remove)

@st.cache_data
def get_hostile_areas():
    """Define hostile areas to avoid (cached)"""
    return [
        # Enemy patrol area (northwest)
        Polygon([
            (-86.81000, 36.14300),
            (-86.80600, 36.14300),
            (-86.80600, 36.14600),
            (-86.81000, 36.14600)
        ]),
        # Adversary stronghold (central area)
        Polygon([
            (-86.80650, 36.14500),
            (-86.80350, 36.14500),
            (-86.80350, 36.14750),
            (-86.80650, 36.14750)
        ]),
        # Sniper overwatch position (northeast)
        Polygon([
            (-86.80400, 36.14750),
            (-86.80150, 36.14750),
            (-86.80150, 36.14950),
            (-86.80400, 36.14950)
        ]),
        # Hostile checkpoint (south)
        Polygon([
            (-86.80800, 36.14200),
            (-86.80400, 36.14200),
            (-86.80400, 36.14400),
            (-86.80800, 36.14400)
        ]),
        # High-risk corridor (east-west through middle)
        Polygon([
            (-86.80900, 36.14600),
            (-86.80200, 36.14600),
            (-86.80200, 36.14700),
            (-86.80900, 36.14700)
        ])
    ]

@st.cache_data
def calculate_hostile_area_intersections(_G_edges, _hostile_areas):
    """Pre-calculate which edges intersect hostile areas and by how much (cached)"""
    edge_penalties = {}
    for u, v, key, data in _G_edges:
        if 'geometry' in data:
            edge_geom = data['geometry']
        else:
            u_point = (data['u_x'], data['u_y'])
            v_point = (data['v_x'], data['v_y'])
            edge_geom = LineString([u_point, v_point])
        total_penalty_length = 0
        for hostile_area in _hostile_areas:
            if edge_geom.intersects(hostile_area):
                try:
                    intersection = edge_geom.intersection(hostile_area)
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

def calculate_route_hostile_exposure(G, route, hostile_areas):
    """Calculate total length a route spends in hostile areas"""
    total_hostile_length = 0
    
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        
        # Get edge geometry
        edge_data = G[u][v][0]  # Take first edge if multiple
        if 'geometry' in edge_data:
            edge_geom = edge_data['geometry']
        else:
            u_point = (G.nodes[u]['x'], G.nodes[u]['y'])
            v_point = (G.nodes[v]['x'], G.nodes[v]['y'])
            edge_geom = LineString([u_point, v_point])
        
        # Check intersection with each hostile area
        for hostile_area in hostile_areas:
            if edge_geom.intersects(hostile_area):
                try:
                    intersection = edge_geom.intersection(hostile_area)
                    if hasattr(intersection, 'length'):
                        # Convert to meters (approximate)
                        intersection_length_meters = intersection.length * 111000
                        total_hostile_length += intersection_length_meters
                except:
                    # Fallback: use partial edge length if intersection calculation fails
                    edge_length_meters = edge_data.get('original_length', 100)
                    total_hostile_length += edge_length_meters * 0.5
    
    return total_hostile_length

def apply_hostile_penalties(G, edge_penalties, hostile_penalty_per_meter):
    """Apply current hostile area penalties to graph (fast operation)"""
    G_copy = G.copy()
    penalties_applied = 0
    for (u, v, key), penalty_length in edge_penalties.items():
        if G_copy.has_edge(u, v):
            original_length = G_copy[u][v][key]['original_length']
            penalty = penalty_length * hostile_penalty_per_meter
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
    st.session_state.start_point = [-86.80867561015874, 36.14420758854631]  # Starting position
if 'end_point' not in st.session_state:
    st.session_state.end_point = [-86.80282365449781, 36.14819247137839]   # Objective
if 'selected_route' not in st.session_state:
    st.session_state.selected_route = None

# Load base graph (only runs once due to caching)
try:
    with st.spinner("Loading tactical map and threat data (one-time setup)..."):
        base_G, threat_data, removed_edges = load_base_graph()
        hostile_areas = get_hostile_areas()
        # Pre-calculate hostile area intersections (also cached)
        G_edges_data = []
        for u, v, key, data in base_G.edges(keys=True, data=True):
            edge_data = data.copy()
            edge_data['u_x'] = base_G.nodes[u]['x']
            edge_data['u_y'] = base_G.nodes[u]['y']
            edge_data['v_x'] = base_G.nodes[v]['x']
            edge_data['v_y'] = base_G.nodes[v]['y']
            G_edges_data.append((u, v, key, edge_data))
        edge_penalties = calculate_hostile_area_intersections(G_edges_data, hostile_areas)
    st.sidebar.success(f"Map loaded! 🚫 {removed_edges} blocked routes from {len(threat_data)} threats")
    st.sidebar.info(f"⚠️ {len(edge_penalties)} routes through hostile areas")
    # Display threat summary
    if threat_data:
        threat_classes = {}
        for threat in threat_data:
            threat_class = threat['class']
            if threat_class in threat_classes:
                threat_classes[threat_class] += 1
            else:
                threat_classes[threat_class] = 1
        st.sidebar.write("**Threat Assessment:**")
        for threat_class, count in threat_classes.items():
            st.sidebar.write(f"• {threat_class}: {count}")
except Exception as e:
    st.error(f"Error loading tactical map: {str(e)}")
    st.stop()

# Create columns for layout
col1, col2 = st.columns([3, 1])

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
        popup="Start Position",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        [st.session_state.end_point[1], st.session_state.end_point[0]],
        popup="Objective",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # Add threat zones if enabled
    if show_blockages:
        for i, threat in enumerate(threat_data):
            threat_polygon = threat['polygon']
            coords = [(lat, lon) for lon, lat in threat_polygon.exterior.coords]
            folium.Polygon(
                coords,
                color='red',
                weight=2,
                fillColor='red',
                fillOpacity=0.5,
                popup=f'💣 {threat["stem"]} ({threat["class"]}) - ID: {threat["class_id"]}'
            ).add_to(m)
    
    # Add hostile areas if enabled
    if show_avoid_zones:
        area_names = ['Enemy Patrol Area', 'Adversary Stronghold', 'Sniper Overwatch', 'Hostile Checkpoint', 'High-Risk Corridor']
        for i, hostile_area in enumerate(hostile_areas):
            coords = [(lat, lon) for lon, lat in hostile_area.exterior.coords]
            folium.Polygon(
                coords,
                color='orange',
                weight=2,
                fillColor='yellow',
                fillOpacity=0.2,
                popup=f'⚠️ {area_names[i]} (Risk penalty: {avoid_penalty}/m)'
            ).add_to(m)
    
    # Calculate routes button
    if st.sidebar.button("Calculate Routes", type="primary"):
        with st.spinner("Calculating tactical routes..."):
            try:
                # Apply current hostile penalties (fast operation)
                G_with_penalties, penalties_applied = apply_hostile_penalties(base_G, edge_penalties, avoid_penalty)
                st.write(f"Start coordinates: {st.session_state.start_point}")
                st.write(f"End coordinates: {st.session_state.end_point}")
                
                # Convert points to nodes with better error handling
                start_lon, start_lat = st.session_state.start_point[0], st.session_state.start_point[1]
                end_lon, end_lat = st.session_state.end_point[0], st.session_state.end_point[1]
                
                try:
                    start_node = ox.distance.nearest_nodes(G_with_penalties, start_lon, start_lat)
                except Exception as e:
                    st.error(f"Error finding start node: {str(e)}")
                    raise
                
                try:
                    end_node = ox.distance.nearest_nodes(G_with_penalties, end_lon, end_lat)
                except Exception as e:
                    st.error(f"Error finding end node: {str(e)}")
                    raise
                
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
                            popup=f'🛣️ Route {i+1} ({len(route)} waypoints)'
                        ).add_to(m)
                    
                    st.session_state.routes = routes
                    st.session_state.route_colors = colors
                    st.session_state.current_graph = G_with_penalties
                    st.session_state.selected_route = None  # Reset selection
                    st.success(f"Found {len(routes)} alternate routes! (Applied {penalties_applied} risk penalties)")
                else:
                    st.error("No viable routes found!")
            except nx.NetworkXNoPath:
                st.error("No path available - objective may be isolated by threats")
            except Exception as e:
                st.error(f"Error calculating routes: {type(e).__name__}: {str(e)}")
                import traceback
                st.text("Full traceback:")
                st.text(traceback.format_exc())
    
    # Display existing routes if they exist
    if 'routes' in st.session_state and 'current_graph' in st.session_state:
        for i, route in enumerate(st.session_state.routes):
            coords = [(st.session_state.current_graph.nodes[node]['y'],
                      st.session_state.current_graph.nodes[node]['x']) for node in route]
            # Highlight selected route with thicker line
            weight = 6 if st.session_state.selected_route == i else 4
            opacity = 1.0 if st.session_state.selected_route == i else 0.6
            folium.PolyLine(
                coords,
                color=st.session_state.route_colors[i],
                weight=weight,
                opacity=opacity,
                popup=f'🛣️ Route {i+1} ({len(route)} waypoints)'
            ).add_to(m)
    
    # Display the interactive map
    map_data = st_folium(m, width=1000, height=700, returned_objects=["last_clicked"])
    
    # Handle map clicks to set new start/end points
    if map_data['last_clicked']:
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lng = map_data['last_clicked']['lng']
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Set as Start Position"):
                st.session_state.start_point = [clicked_lng, clicked_lat]
                if 'routes' in st.session_state:
                    del st.session_state.routes
                st.rerun()
        with col_b:
            if st.button("Set as Objective"):
                st.session_state.end_point = [clicked_lng, clicked_lat]
                if 'routes' in st.session_state:
                    del st.session_state.routes
                st.rerun()

with col2:
    st.subheader("Mission Parameters")
    
    # Display current points
    st.write(f"Start: ({st.session_state.start_point[1]:.6f} N, {st.session_state.start_point[0]:.6f} E)")
    
    st.write(f"Target: ({st.session_state.end_point[1]:.6f} N, {st.session_state.end_point[0]:.6f} E)")
    
    # Route statistics and selection
    if 'routes' in st.session_state and 'current_graph' in st.session_state:
        st.write("**Route Analysis:**")
        G_current = st.session_state.current_graph
        
        for i, route in enumerate(st.session_state.routes):
            # Calculate total route length
            total_length = 0
            for j in range(len(route) - 1):
                u, v = route[j], route[j+1]
                edge_data = G_current[u][v][0]  # Take first edge if multiple
                total_length += edge_data.get('length', 0)
            
            # Calculate hostile area exposure
            hostile_exposure = calculate_route_hostile_exposure(G_current, route, hostile_areas)
            
            # Create selectable button for each route with color indicator
            is_selected = st.session_state.selected_route == i
            route_color = st.session_state.route_colors[i]
            
            # Create a colored box using HTML
            color_box = f'<span style="display:inline-block; width:20px; height:20px; background-color:{route_color}; border:1px solid #000; margin-right:8px; vertical-align:middle;"></span>'
            button_label = f"{'✓ ' if is_selected else ''}Route {i+1}"
            
            # Display color indicator and button
            col_color, col_button = st.columns([0.3, 2])
            with col_color:
                st.markdown(color_box, unsafe_allow_html=True)
            with col_button:
                if st.button(button_label, key=f"route_{i}", use_container_width=True):
                    st.session_state.selected_route = i
                    # Print to console
                    print(f"\n{'='*60}")
                    print(f"SELECTED ROUTE {i+1}")
                    print(f"{'='*60}")
                    print(f"Number of waypoints: {len(route)}")
                    print(f"Total length: {total_length:.2f} meters")
                    print(f"Hostile exposure: {hostile_exposure:.2f} meters")
                    print(f"\nRoute nodes: {route}")
                    print(f"\nRoute coordinates (lat, lon):")
                    for j, node in enumerate(route):
                        lat = G_current.nodes[node]['y']
                        lon = G_current.nodes[node]['x']
                        print(f"  Waypoint {j+1}: ({lat:.6f}, {lon:.6f})")
                    print(f"{'='*60}\n")
                    st.rerun()
            
            st.write(f"  {len(route)} waypoints, {total_length:.0f}m, Hostile: {hostile_exposure:.0f}m")
        
        # Display selected route details
        if st.session_state.selected_route is not None:
            st.divider()
            st.subheader(f"Selected: Route {st.session_state.selected_route + 1}")
            selected_route = st.session_state.routes[st.session_state.selected_route]
            
            # Calculate details
            total_length = sum(
                G_current[selected_route[j]][selected_route[j+1]][0].get('length', 0)
                for j in range(len(selected_route) - 1)
            )
            hostile_exposure = calculate_route_hostile_exposure(G_current, selected_route, hostile_areas)
            
            st.write(f"**Waypoints:** {len(selected_route)}")
            st.write(f"**Total Distance:** {total_length:.2f}m")
            st.write(f"**Hostile Exposure:** {hostile_exposure:.2f}m")
            st.write(f"**Risk Ratio:** {(hostile_exposure/total_length*100):.1f}%")
