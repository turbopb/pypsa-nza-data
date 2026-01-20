# -*- coding: utf-8 -*-
"""
nza_transmission_line_processo.py

Transmission line distance calculation using graph theory

@author: Phil
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from pyproj import Geod
import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import re



def convert_to_linestring(geometry_data):
    """Convert GeoJSON geometry to single LineString."""
    geom_type = geometry_data['type']
    coords = geometry_data['coordinates']
    
    if geom_type == 'LineString':
        return LineString(coords)
    elif geom_type == 'MultiLineString':
        all_coords = []
        for linestring in coords:
            all_coords.extend(linestring)
        return LineString(all_coords)
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")


def calculate_linestring_length(linestring):
    """Calculate geodesic length of a LineString in km."""
    geod = Geod(ellps='WGS84')
    coords = list(linestring.coords)
    
    total_length = 0.0
    for i in range(len(coords) - 1):
        _, _, distance = geod.inv(
            coords[i][0], coords[i][1],
            coords[i+1][0], coords[i+1][1]
        )
        total_length += distance
    
    return total_length / 1000  # km


def extract_substations_from_geojson(geojson_path):
    """
    Extract all unique substation codes from Transpower GeoJSON MXLOCATION field.
    
    MXLOCATION format is typically: "XXX-YYY-A" where XXX and YYY are substation codes.
    
    Returns:
    --------
    set: All unique 3-letter substation codes found
    """
    with open(geojson_path, 'r') as f:
        line_data = json.load(f)
    
    substations = set()
    
    for feature in line_data['features']:
        mxlocation = feature['properties']['MXLOCATION']
        
        # Parse MXLOCATION - typically format is "XXX-YYY-A" or "XXX-YYY-A-CBL"
        # Extract 3-letter codes before hyphens
        parts = mxlocation.split('-')
        
        for part in parts:
            # Look for 3-letter uppercase codes (typical substation format)
            if len(part) == 3 and part.isupper() and part.isalpha():
                substations.add(part)
    
    return substations


def compare_substations(geojson_path, bus_csv_path):
    """
    Compare substations in GeoJSON vs. sites.csv to find missing ones.
    """
    # Extract from GeoJSON
    geojson_subs = extract_substations_from_geojson(geojson_path)
    
    # Load from sites.csv
    buses_df = pd.read_csv(bus_csv_path)
    csv_subs = set(buses_df['site'].tolist())
    
    # Find differences
    missing_from_csv = geojson_subs - csv_subs
    extra_in_csv = csv_subs - geojson_subs
    
    print("\n" + "="*80)
    print("SUBSTATION COMPARISON: GeoJSON vs. sites.csv")
    print("="*80)
    print(f"\nSubstations in GeoJSON MXLOCATION fields: {len(geojson_subs)}")
    print(f"Substations in sites.csv: {len(csv_subs)}")
    print(f"Missing from sites.csv: {len(missing_from_csv)}")
    print(f"Extra in sites.csv (not in GeoJSON): {len(extra_in_csv)}")
    
    if missing_from_csv:
        print(f"\n\nMISSING SUBSTATIONS (in GeoJSON but NOT in sites.csv): {len(missing_from_csv)}")
        print("These substations appear in Transpower data but are missing from your sites.csv:")
        print(sorted(missing_from_csv))
    
    if extra_in_csv:
        print(f"\n\nEXTRA SUBSTATIONS (in sites.csv but NOT in GeoJSON): {len(extra_in_csv)}")
        print("These are in your sites.csv but don't appear in Transpower line names:")
        print(sorted(extra_in_csv))
    
    return {
        'geojson_substations': geojson_subs,
        'csv_substations': csv_subs,
        'missing_from_csv': missing_from_csv,
        'extra_in_csv': extra_in_csv
    }


def diagnose_unmatched_lines(geojson_path, bus_csv_path, max_distance_m=1000):
    """
    Diagnose why lines aren't matching to substations.
    """
    # Load data
    buses_df = pd.read_csv(bus_csv_path)
    bus_geometries = [Point(lon, lat) for lon, lat in 
                      zip(buses_df['long'], buses_df['lat'])]
    bus_gdf = gpd.GeoDataFrame(buses_df, geometry=bus_geometries, crs='EPSG:4326')
    
    with open(geojson_path, 'r') as f:
        line_data = json.load(f)
    
    geod = Geod(ellps='WGS84')
    
    print("\n" + "="*80)
    print("DIAGNOSING UNMATCHED LINES")
    print("="*80)
    
    unmatched_details = []
    
    for feature in line_data['features']:
        mxlocation = feature['properties']['MXLOCATION']
        
        try:
            line_geom = convert_to_linestring(feature['geometry'])
            coords = list(line_geom.coords)
            start_point = Point(coords[0])
            end_point = Point(coords[-1])
            
            # Find nearest buses with ANY distance
            start_distances = []
            end_distances = []
            
            for idx, row in bus_gdf.iterrows():
                # Distance to start
                _, _, dist_start = geod.inv(
                    start_point.x, start_point.y,
                    row.geometry.x, row.geometry.y
                )
                start_distances.append((row['site'], dist_start))
                
                # Distance to end
                _, _, dist_end = geod.inv(
                    end_point.x, end_point.y,
                    row.geometry.x, row.geometry.y
                )
                end_distances.append((row['site'], dist_end))
            
            # Sort by distance
            start_distances.sort(key=lambda x: x[1])
            end_distances.sort(key=lambda x: x[1])
            
            # Get closest
            nearest_start = start_distances[0]
            nearest_end = end_distances[0]
            
            # Check if within threshold
            start_matched = nearest_start[1] <= max_distance_m
            end_matched = nearest_end[1] <= max_distance_m
            
            if not start_matched or not end_matched or nearest_start[0] == nearest_end[0]:
                unmatched_details.append({
                    'mxlocation': mxlocation,
                    'nearest_start': nearest_start[0],
                    'dist_start_m': nearest_start[1],
                    'start_ok': start_matched,
                    'nearest_end': nearest_end[0],
                    'dist_end_m': nearest_end[1],
                    'end_ok': end_matched,
                    'self_loop': nearest_start[0] == nearest_end[0],
                    'reason': 'SELF-LOOP' if nearest_start[0] == nearest_end[0] 
                             else 'START TOO FAR' if not start_matched 
                             else 'END TOO FAR' if not end_matched
                             else 'BOTH TOO FAR'
                })
        
        except Exception as e:
            continue
    
    # Convert to DataFrame and display
    df = pd.DataFrame(unmatched_details)
    
    if len(df) > 0:
        print(f"\nFound {len(df)} unmatched lines:\n")
        
        # Group by reason
        print("BREAKDOWN BY REASON:")
        print(df['reason'].value_counts())
        
        print("\n\nFIRST 20 UNMATCHED LINES:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df[['mxlocation', 'nearest_start', 'dist_start_m', 'nearest_end', 'dist_end_m', 'reason']].head(20).to_string(index=False))
        
        # Find lines where both ends are far
        both_far = df[(df['dist_start_m'] > max_distance_m) & (df['dist_end_m'] > max_distance_m)]
        if len(both_far) > 0:
            print(f"\n\nLINES WITH BOTH ENDS FAR FROM ANY SUBSTATION ({len(both_far)}):")
            print("These substations are probably MISSING from sites.csv:")
            for idx, row in both_far.head(10).iterrows():
                print(f"  {row['mxlocation']}: nearest to start={row['nearest_start']} ({row['dist_start_m']/1000:.1f} km), "
                      f"nearest to end={row['nearest_end']} ({row['dist_end_m']/1000:.1f} km)")
        
        # Find self-loops
        self_loops = df[df['self_loop']]
        if len(self_loops) > 0:
            print(f"\n\nSELF-LOOPS ({len(self_loops)}):")
            print("Both endpoints match to the SAME substation:")
            for idx, row in self_loops.head(10).iterrows():
                print(f"  {row['mxlocation']}: both ends -> {row['nearest_start']} "
                      f"(distances: {row['dist_start_m']:.0f}m, {row['dist_end_m']:.0f}m)")
        
        return df
    else:
        print("All lines matched successfully!")
        return None


def find_nearest_bus(point, bus_gdf, max_distance_m=1000):
    """
    Find the nearest bus to a point within threshold.
    
    Returns:
    --------
    tuple: (bus_code, distance_m) or (None, None)
    """
    geod = Geod(ellps='WGS84')
    
    min_distance = float('inf')
    nearest_bus = None
    
    for idx, row in bus_gdf.iterrows():
        _, _, distance = geod.inv(
            point.x, point.y,
            row.geometry.x, row.geometry.y
        )
        
        if distance < min_distance:
            min_distance = distance
            nearest_bus = row['site']
    
    if min_distance <= max_distance_m:
        return nearest_bus, min_distance
    else:
        return None, None


def build_transmission_network(geojson_path, bus_csv_path, 
                               max_endpoint_distance_m=1000):
    """
    Build a NetworkX graph of the transmission network.
    
    Returns:
    --------
    tuple: (G, line_data_dict, bus_gdf)
        - G: NetworkX MultiGraph with transmission lines as edges
        - line_data_dict: {(bus_a, bus_b, line_id): {geometry, metadata}}
        - bus_gdf: GeoDataFrame of buses
    """
    # Load buses
    buses_df = pd.read_csv(bus_csv_path)
    bus_geometries = [Point(lon, lat) for lon, lat in 
                      zip(buses_df['long'], buses_df['lat'])]
    bus_gdf = gpd.GeoDataFrame(buses_df, geometry=bus_geometries, crs='EPSG:4326')
    
    # Load transmission lines
    with open(geojson_path, 'r') as f:
        line_data = json.load(f)
    
    # Create MultiGraph (allows multiple edges between same nodes)
    G = nx.MultiGraph()
    line_data_dict = {}
    
    print("\n" + "="*80)
    print("BUILDING TRANSMISSION NETWORK GRAPH")
    print("="*80)
    print(f"Processing {len(line_data['features'])} transmission lines...")
    
    unmatched_lines = []
    matched_count = 0
    
    for feature in line_data['features']:
        mxlocation = feature['properties']['MXLOCATION']
        
        try:
            # Convert to LineString
            line_geom = convert_to_linestring(feature['geometry'])
            
            # Get endpoints
            coords = list(line_geom.coords)
            start_point = Point(coords[0])
            end_point = Point(coords[-1])
            
            # Find nearest buses to endpoints
            bus_start, dist_start = find_nearest_bus(start_point, bus_gdf, max_endpoint_distance_m)
            bus_end, dist_end = find_nearest_bus(end_point, bus_gdf, max_endpoint_distance_m)
            
            if bus_start and bus_end and bus_start != bus_end:
                # Calculate line length
                length_km = calculate_linestring_length(line_geom)
                
                # Add edge to graph (MultiGraph allows multiple edges between same nodes)
                edge_key = G.add_edge(
                    bus_start, 
                    bus_end,
                    mxlocation=mxlocation,
                    length_km=length_km,
                    geometry=line_geom,
                    description=feature['properties'].get('description', ''),
                    voltage=feature['properties'].get('designvolt', ''),
                    status=feature['properties'].get('status', '')
                )
                
                # Store detailed line data
                line_data_dict[(bus_start, bus_end, edge_key)] = {
                    'mxlocation': mxlocation,
                    'length_km': length_km,
                    'geometry': line_geom,
                    'description': feature['properties'].get('description', ''),
                    'voltage': feature['properties'].get('designvolt', ''),
                    'start_match_distance_m': dist_start,
                    'end_match_distance_m': dist_end
                }
                
                matched_count += 1
                if matched_count <= 20:  # Only print first 20
                    print(f"  {mxlocation}: {bus_start} <-> {bus_end} ({length_km:.2f} km)")
            else:
                unmatched_lines.append({
                    'mxlocation': mxlocation,
                    'bus_start': bus_start,
                    'bus_end': bus_end,
                    'dist_start': dist_start if bus_start else None,
                    'dist_end': dist_end if bus_end else None
                })
        
        except Exception as e:
            print(f"  Error processing {mxlocation}: {e}")
            continue
    
    if matched_count > 20:
        print(f"  ... and {matched_count - 20} more lines")
    
    print(f"\nNetwork built:")
    print(f"  Nodes (substations): {G.number_of_nodes()}")
    print(f"  Edges (transmission lines): {G.number_of_edges()}")
    print(f"  Matched: {matched_count}")
    print(f"  Unmatched: {len(unmatched_lines)}")
    
    return G, line_data_dict, bus_gdf


def find_all_connecting_lines(bus_a, bus_b, G, line_data_dict):
    """
    Find all transmission lines that directly connect two buses.
    
    Returns:
    --------
    list: List of dicts containing line information
    """
    results = []
    
    # Check if both buses exist in graph
    if bus_a not in G or bus_b not in G:
        return results
    
    # Get all edges between these two nodes
    # In MultiGraph, there can be multiple edges between same nodes
    if G.has_edge(bus_a, bus_b):
        # Get all edge keys for this bus pair
        edge_data = G[bus_a][bus_b]
        
        for edge_key, edge_attrs in edge_data.items():
            # Find the corresponding entry in line_data_dict
            # Need to check both (bus_a, bus_b) and (bus_b, bus_a) since graph is undirected
            line_key = (bus_a, bus_b, edge_key)
            if line_key not in line_data_dict:
                line_key = (bus_b, bus_a, edge_key)
            
            if line_key in line_data_dict:
                line_info = line_data_dict[line_key].copy()
                results.append(line_info)
    
    return results


def calculate_bus_pair_distances(bus_pairs, G, line_data_dict):
    """
    Calculate distances for all bus pairs.
    
    Returns:
    --------
    pd.DataFrame: Results with all connecting lines
    """
    results = []
    
    print("\n" + "="*80)
    print("CALCULATING DISTANCES FOR BUS PAIRS")
    print("="*80)
    
    for i, (bus_a, bus_b) in enumerate(bus_pairs):
        connecting_lines = find_all_connecting_lines(bus_a, bus_b, G, line_data_dict)
        
        if connecting_lines:
            for j, line in enumerate(connecting_lines):
                results.append({
                    'bus_a': bus_a,
                    'bus_b': bus_b,
                    'mxlocation': line['mxlocation'],
                    'description': line['description'],
                    'voltage': line['voltage'],
                    'length_km': line['length_km'],
                    'start_match_distance_m': line['start_match_distance_m'],
                    'end_match_distance_m': line['end_match_distance_m'],
                    'line_number': j + 1,
                    'total_lines': len(connecting_lines)
                })
                
                if i < 10:  # Print first 10
                    print(f"  {bus_a} <-> {bus_b}: {line['mxlocation']} ({line['length_km']:.2f} km)")
                    if len(connecting_lines) > 1:
                        print(f"    [Line {j+1} of {len(connecting_lines)}]")
        else:
            # No direct connection found
            if i < 10:
                print(f"  {bus_a} <-> {bus_b}: NO DIRECT CONNECTION")
            results.append({
                'bus_a': bus_a,
                'bus_b': bus_b,
                'mxlocation': None,
                'description': 'NO DIRECT CONNECTION',
                'voltage': None,
                'length_km': None,
                'start_match_distance_m': None,
                'end_match_distance_m': None,
                'line_number': 0,
                'total_lines': 0
            })
    
    if len(bus_pairs) > 10:
        print(f"  ... and {len(bus_pairs) - 10} more bus pairs")
    
    return pd.DataFrame(results)


def analyze_network_connectivity(G):
    """Print network statistics and identify key nodes."""
    print("\n" + "="*80)
    print("NETWORK ANALYSIS")
    print("="*80)
    
    # Node degree (how many lines connect to each substation)
    degrees = dict(G.degree())
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 most connected substations:")
    for node, degree in sorted_degrees[:10]:
        print(f"  {node}: {degree} connections")
    
    # Check for buses with multiple parallel lines
    print("\nBus pairs with multiple parallel transmission lines:")
    multi_lines = []
    for u, v in G.edges():
        if len(G[u][v]) > 1 and (u, v) not in [(mv, mu) for mu, mv in multi_lines]:
            multi_lines.append((u, v))
            print(f"  {u} <-> {v}: {len(G[u][v])} parallel lines")
            for key, data in G[u][v].items():
                print(f"    - {data['mxlocation']} ({data['length_km']:.2f} km)")


def plot_transpower_network(G, bus_gdf, line_data_dict, output_dir):
    """
    Plot the FULL Transpower network (auto-detected from GeoJSON).
    Shows plots live in Spyder AND saves to PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("PLOTTING FULL TRANSPOWER NETWORK")
    print("="*80)
    
    # Voltage colors
    voltage_colors = {
        '220': '#d62728',
        '110': '#ff7f0e',
        '66': '#2ca02c',
        '50': '#1f77b4',
        '33': '#9467bd',
        '11': '#8c564b',
    }
    
    # Filter bus_gdf to only include buses in the graph
    graph_buses = set(G.nodes())
    graph_bus_gdf = bus_gdf[bus_gdf['site'].isin(graph_buses)].copy()
    
    # ========== PLOT 1: Schematic ==========
    fig, ax = plt.subplots(figsize=(24, 20))
    
    # Create position dictionary using actual lat/lon coordinates
    pos = {}
    for idx, row in graph_bus_gdf.iterrows():
        pos[row['site']] = (row.geometry.x, row.geometry.y)
    
    # Draw edges with color coding by voltage
    for u, v, key, data in G.edges(keys=True, data=True):
        voltage = data.get('voltage', '')
        color = voltage_colors.get(voltage, '#7f7f7f')
        
        num_lines = len(G[u][v])
        width = 2.0 if num_lines > 1 else 1.0
        alpha = 0.7 if num_lines > 1 else 0.5
        
        # Draw edge
        if u in pos and v in pos:
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, color=color, linewidth=width, alpha=alpha, zorder=1)
    
    # Draw nodes
    node_degrees = dict(G.degree())
    for node in G.nodes():
        if node in pos:
            degree = node_degrees.get(node, 1)
            size = min(degree * 100, 1000)
            ax.scatter(pos[node][0], pos[node][1], 
                      s=size, c='lightblue', edgecolors='black', 
                      linewidths=1, zorder=5)
            
            # Label high-degree nodes
            if degree >= 3:
                ax.annotate(node, pos[node],
                           fontsize=7, fontweight='bold',
                           xytext=(5, 5), textcoords='offset points')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=voltage_colors['220'], linewidth=2, label='220 kV'),
        Line2D([0], [0], color=voltage_colors['110'], linewidth=2, label='110 kV'),
        Line2D([0], [0], color=voltage_colors['66'], linewidth=2, label='66 kV'),
        Line2D([0], [0], color=voltage_colors['50'], linewidth=2, label='50 kV'),
        Line2D([0], [0], color='#7f7f7f', linewidth=2, label='Other'),
        Line2D([0], [0], color='black', linewidth=2, alpha=0.7, label='Parallel lines'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title(f'FULL Transpower Network - Schematic\n({G.number_of_nodes()} substations, {G.number_of_edges()} lines)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'transpower_full_schematic.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.show()  # Show live in Spyder
    
    # ========== PLOT 2: Detailed with Actual Geometries ==========
    fig, ax = plt.subplots(figsize=(24, 20))
    
    for (bus_a, bus_b, edge_key), data in line_data_dict.items():
        geom = data['geometry']
        voltage = data.get('voltage', '')
        color = voltage_colors.get(voltage, '#7f7f7f')
        
        num_lines = len(G[bus_a][bus_b]) if G.has_edge(bus_a, bus_b) else 1
        linewidth = 1.5 if num_lines > 1 else 1.0
        
        coords = list(geom.coords)
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        ax.plot(lons, lats, color=color, linewidth=linewidth, alpha=0.6)
    
    # Plot substations
    for idx, row in graph_bus_gdf.iterrows():
        degree = G.degree(row['site'])
        size = min(degree * 80, 800)
        
        ax.scatter(row.geometry.x, row.geometry.y, 
                  s=size, c='lightblue', edgecolors='black', 
                  linewidths=1, zorder=5)
        
        if degree >= 3:
            ax.annotate(row['site'], 
                       (row.geometry.x, row.geometry.y),
                       fontsize=7, fontweight='bold',
                       xytext=(5, 5), textcoords='offset points')
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.set_title(f'FULL Transpower Network - Detailed Geometry\n(Actual transmission line routes)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'transpower_full_detailed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.show()  # Show live in Spyder


def plot_user_specified_network(bus_pairs_df, results_df, bus_gdf, line_data_dict, 
                                output_dir, G):
    """
    Plot the USER's specified network from lines_data.csv.
    Shows plots live in Spyder AND saves to PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Voltage colors
    voltage_colors = {
        '220': '#d62728',
        '110': '#ff7f0e',
        '66': '#2ca02c',
        '50': '#1f77b4',
        '33': '#9467bd',
        '11': '#8c564b',
    }
    
    # Get unique buses from user's bus pairs
    user_buses = set(bus_pairs_df['bus0'].tolist() + bus_pairs_df['bus1'].tolist())
    
    # Filter bus_gdf to only include user's buses
    user_bus_gdf = bus_gdf[bus_gdf['site'].isin(user_buses)].copy()
    
    print("\n" + "="*80)
    print("PLOTTING USER-SPECIFIED NETWORK")
    print("="*80)
    print(f"Buses: {len(user_buses)}, Connections: {len(bus_pairs_df)}")
    
    # ========== PLOT 1: User Network Schematic ==========
    fig, ax = plt.subplots(figsize=(20, 16))
    
    for idx, row in results_df.iterrows():
        if pd.isna(row['mxlocation']):
            continue
        
        bus_a = row['bus_a']
        bus_b = row['bus_b']
        
        # Get bus coordinates
        bus_a_row = user_bus_gdf[user_bus_gdf['site'] == bus_a]
        bus_b_row = user_bus_gdf[user_bus_gdf['site'] == bus_b]
        
        if len(bus_a_row) == 0 or len(bus_b_row) == 0:
            continue
        
        x1, y1 = bus_a_row.iloc[0].geometry.x, bus_a_row.iloc[0].geometry.y
        x2, y2 = bus_b_row.iloc[0].geometry.x, bus_b_row.iloc[0].geometry.y
        
        voltage = str(row['voltage'])
        color = voltage_colors.get(voltage, '#7f7f7f')
        
        linewidth = 2.0 if row['total_lines'] > 1 else 1.0
        
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.6)
    
    # Plot substations
    for idx, row in user_bus_gdf.iterrows():
        degree = sum(1 for bp in bus_pairs_df.itertuples() 
                    if bp.bus0 == row['site'] or bp.bus1 == row['site'])
        size = min(degree * 100, 500)
        
        ax.scatter(row.geometry.x, row.geometry.y, 
                  s=size, c='lightcoral', edgecolors='black', 
                  linewidths=1.5, zorder=5)
        
        # Label all substations
        ax.annotate(row['site'], 
                   (row.geometry.x, row.geometry.y),
                   fontsize=7, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=voltage_colors['220'], linewidth=2, label='220 kV'),
        Line2D([0], [0], color=voltage_colors['110'], linewidth=2, label='110 kV'),
        Line2D([0], [0], color=voltage_colors['66'], linewidth=2, label='66 kV'),
        Line2D([0], [0], color='#7f7f7f', linewidth=2, label='Other'),
        Line2D([0], [0], color='black', linewidth=2, alpha=0.7, label='Parallel lines'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    num_connected = len(results_df[results_df['mxlocation'].notna()])
    ax.set_title(f'USER-SPECIFIED Network - Schematic\n(from lines_data.csv: {len(user_buses)} buses, {num_connected} lines)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'user_network_schematic.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.show()  # Show live in Spyder
    
    # ========== PLOT 2: User Network Detailed ==========
    fig, ax = plt.subplots(figsize=(20, 16))
    
    for idx, row in results_df.iterrows():
        if pd.isna(row['mxlocation']):
            continue
        
        bus_a = row['bus_a']
        bus_b = row['bus_b']
        
        # Find the geometry for this line
        connecting_lines = find_all_connecting_lines(bus_a, bus_b, G, line_data_dict)
        
        for line in connecting_lines:
            if line['mxlocation'] == row['mxlocation']:
                geom = line['geometry']
                voltage = str(line['voltage'])
                color = voltage_colors.get(voltage, '#7f7f7f')
                
                coords = list(geom.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                
                linewidth = 1.5 if row['total_lines'] > 1 else 1.0
                ax.plot(lons, lats, color=color, linewidth=linewidth, alpha=0.7)
                break
    
    # Plot substations
    for idx, row in user_bus_gdf.iterrows():
        degree = sum(1 for bp in bus_pairs_df.itertuples() 
                    if bp.bus0 == row['site'] or bp.bus1 == row['site'])
        size = min(degree * 100, 500)
        
        ax.scatter(row.geometry.x, row.geometry.y, 
                  s=size, c='lightcoral', edgecolors='black', 
                  linewidths=1.5, zorder=5)
        
        ax.annotate(row['site'], 
                   (row.geometry.x, row.geometry.y),
                   fontsize=7, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.set_title(f'USER-SPECIFIED Network - Detailed Geometry\n(Actual transmission line routes)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'user_network_detailed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.show()  # Show live in Spyder


def plot_comparison_overlay(bus_pairs_df, results_df, bus_gdf, line_data_dict, 
                            output_dir, G):
    """
    Plot BOTH networks overlaid to show differences.
    User network in RED, Full Transpower network in BLUE.
    Shows plot live in Spyder AND saves to PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("PLOTTING COMPARISON OVERLAY")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(24, 20))
    
    # Plot FULL Transpower network in blue (background)
    for (bus_a, bus_b, edge_key), data in line_data_dict.items():
        geom = data['geometry']
        coords = list(geom.coords)
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        ax.plot(lons, lats, color='blue', linewidth=0.8, alpha=0.3, zorder=1)
    
    # Plot USER network in red (foreground)
    user_buses = set(bus_pairs_df['bus0'].tolist() + bus_pairs_df['bus1'].tolist())
    user_bus_gdf = bus_gdf[bus_gdf['site'].isin(user_buses)].copy()
    
    for idx, row in results_df.iterrows():
        if pd.isna(row['mxlocation']):
            continue
        
        bus_a = row['bus_a']
        bus_b = row['bus_b']
        
        connecting_lines = find_all_connecting_lines(bus_a, bus_b, G, line_data_dict)
        
        for line in connecting_lines:
            if line['mxlocation'] == row['mxlocation']:
                geom = line['geometry']
                coords = list(geom.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                
                linewidth = 2.0 if row['total_lines'] > 1 else 1.5
                ax.plot(lons, lats, color='red', linewidth=linewidth, alpha=0.7, zorder=2)
                break
    
    # Plot all Transpower substations (small blue)
    graph_buses = set(G.nodes())
    graph_bus_gdf = bus_gdf[bus_gdf['site'].isin(graph_buses)].copy()
    
    for idx, row in graph_bus_gdf.iterrows():
        ax.scatter(row.geometry.x, row.geometry.y, 
                  s=30, c='lightblue', edgecolors='blue', 
                  linewidths=0.5, zorder=3, alpha=0.5)
    
    # Plot user substations (larger red)
    for idx, row in user_bus_gdf.iterrows():
        degree = sum(1 for bp in bus_pairs_df.itertuples() 
                    if bp.bus0 == row['site'] or bp.bus1 == row['site'])
        size = min(degree * 100, 500)
        
        ax.scatter(row.geometry.x, row.geometry.y, 
                  s=size, c='lightcoral', edgecolors='red', 
                  linewidths=2, zorder=5)
        
        ax.annotate(row['site'], 
                   (row.geometry.x, row.geometry.y),
                   fontsize=8, fontweight='bold', color='red',
                   xytext=(5, 5), textcoords='offset points')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=1, alpha=0.5, label=f'Transpower Full ({G.number_of_edges()} lines)'),
        Line2D([0], [0], color='red', linewidth=2, label=f'Your Network ({len(results_df[results_df["mxlocation"].notna()])} lines)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=8, label=f'Transpower Substations ({G.number_of_nodes()})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
               markersize=10, label=f'Your Substations ({len(user_buses)})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    ax.set_title('COMPARISON: Full Transpower (Blue) vs Your Network (Red)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comparison_overlay.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.show()  # Show live in Spyder


# Main execution
if __name__ == "__main__":
    ROOT = 'C:/Users/Public/Documents/Thesis/analysis/PYPSA-NZA/'
    
    geojson_path = ROOT + 'data/external/static/transmission_lines.geojson'
    bus_csv_path = ROOT + 'data/processed/static/sites.csv'
    network_plots_dir = ROOT + 'network_plots'
    
    # ========== STEP 1: Compare substations ==========
    comparison = compare_substations(geojson_path, bus_csv_path)
    
    # Save missing substations to file
    if comparison['missing_from_csv']:
        missing_df = pd.DataFrame(list(comparison['missing_from_csv']), columns=['substation_code'])
        missing_path = os.path.join(network_plots_dir, 'missing_substations.csv')
        os.makedirs(network_plots_dir, exist_ok=True)
        missing_df.to_csv(missing_path, index=False)
        print(f"\nMissing substations saved to: {missing_path}")
    
    # ========== STEP 2: Diagnose unmatched lines ==========
    unmatched_df = diagnose_unmatched_lines(geojson_path, bus_csv_path, max_distance_m=1000)
    
    if unmatched_df is not None:
        unmatched_path = os.path.join(network_plots_dir, 'unmatched_lines_diagnosis.csv')
        unmatched_df.to_csv(unmatched_path, index=False)
        print(f"\nUnmatched lines diagnostic saved to: {unmatched_path}")
    
    # ========== STEP 3: Build the network graph ==========
    G, line_data_dict, bus_gdf = build_transmission_network(
        geojson_path=geojson_path,
        bus_csv_path=bus_csv_path,
        max_endpoint_distance_m=1000
    )
    
    # ========== STEP 4: Analyze network ==========
    analyze_network_connectivity(G)
    
    # ========== STEP 5: Load user's bus pairs ==========
    lines_csv_path = ROOT + 'data/processed/temp_dump/static/lines_data.csv'
    lines_df = pd.read_csv(lines_csv_path)
    bus_pairs = list(zip(lines_df['bus0'], lines_df['bus1']))
    
    print(f"\nLoaded {len(bus_pairs)} bus pairs from lines_data.csv")
    
    # ========== STEP 6: Calculate distances ==========
    results_df = calculate_bus_pair_distances(bus_pairs, G, line_data_dict)
    
    # Save results
    output_path = ROOT + 'data/processed/temp_dump/static/bus_pair_distances_network.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # ========== STEP 7: Create plots ==========
    # Plot FULL Transpower network
    plot_transpower_network(G, bus_gdf, line_data_dict, network_plots_dir)
    
    # Plot USER'S specified network
    plot_user_specified_network(lines_df, results_df, bus_gdf, line_data_dict, 
                                network_plots_dir, G)
    
    # Plot COMPARISON overlay
    plot_comparison_overlay(lines_df, results_df, bus_gdf, line_data_dict,
                            network_plots_dir, G)
    
    # ========== STEP 8: Final Summary ==========
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nFULL Transpower Network:")
    print(f"  Substations in GeoJSON: {len(comparison['geojson_substations'])}")
    print(f"  Substations in sites.csv: {len(comparison['csv_substations'])}")
    print(f"  Missing from sites.csv: {len(comparison['missing_from_csv'])}")
    print(f"  Matched substations in graph: {G.number_of_nodes()}")
    print(f"  Transmission lines in graph: {G.number_of_edges()}")
    
    print(f"\nYOUR Specified Network:")
    print(f"  Substations: {len(set(lines_df['bus0'].tolist() + lines_df['bus1'].tolist()))}")
    print(f"  Bus pairs requested: {len(bus_pairs)}")
    print(f"  Direct connections found: {len(results_df[results_df['mxlocation'].notna()])}")
    print(f"  Missing connections: {len(results_df[results_df['mxlocation'].isna()])}")
    print(f"  Parallel lines: {len(results_df[results_df['total_lines'] > 1].drop_duplicates(['bus_a', 'bus_b']))}")
    
    print(f"\nAll output saved to: {network_plots_dir}")
    print("\nDone!")