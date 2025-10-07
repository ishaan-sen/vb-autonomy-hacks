"""
Autonomous Drone Path Planning for Military Convoy Escort Operations

This module implements a multi-agent drone path planning system for convoy protection
using formation control and perimeter security strategies.

Mathematical Framework:
    - Formation Control: Maintains rigid geometric formations using leader-follower approach
    - Patrol Optimization: Implements oscillatory patrol patterns for maximum area coverage
    - Circular Perimeter Defense: Distributes drones uniformly on protective circle

Algorithm Overview:
    1. Leader Path Generation: Parametric curve generation for convoy trajectory
    2. Formation Control: Rigid body transformation with offset vectors
    3. Patrol Pattern: Bidirectional sweeping with dynamic boundary constraints
    4. Perimeter Security: Circular formation with angular velocity synchronization

References:
    [1] Beard, R. W., & McLain, T. W. (2012). Small Unmanned Aircraft: Theory and Practice.
        Princeton University Press.
    [2] Oh, K. K., Park, M. C., & Ahn, H. S. (2015). A survey of multi-agent formation control.
        Automatica, 53, 424-440.
    [3] Nigam, N., & Bieniawski, S. (2011). An analytical approach to multi-vehicle rendezvous.
        AIAA Guidance, Navigation, and Control Conference.
    [4] Shima, T., & Rasmussen, S. (2009). UAV Cooperative Decision and Control: Challenges
        and Practical Approaches. SIAM.

Author: Research Team
Date: October 2025
Version: 1.0
License: MIT
"""

import numpy as np
import json
from typing import List, Tuple, Dict, Any

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================

# Geographic Coordinate System (WGS84)
# Operational Area Boundaries [decimal degrees]
LONGITUDE_MIN = -86.8      # Western boundary
LONGITUDE_MAX = -86.2      # Eastern boundary
LATITUDE_MIN = 36.13       # Southern boundary
LATITUDE_MAX = 37.13       # Northern boundary

# Drone Sensing Parameters
# Field of View (FOV) dimensions in degrees
# Approximation: ~87m at 36°N latitude
FOV_LATITUDE = 0.00078045   # Angular FOV in latitude direction
FOV_LONGITUDE = 0.00128331  # Angular FOV in longitude direction

# Formation Control Parameters
FORMATION_ROWS = 5          # Number of rows in formation matrix
FORMATION_COLS = 4          # Number of columns in formation matrix
FORMATION_SPACING = 0.005   # Inter-agent spacing [degrees]

# Perimeter Defense Parameters
PERIMETER_DRONES = 25       # Number of drones on perimeter
PERIMETER_RADIUS = 0.04     # Radius of protective circle [degrees]

# Speed Multipliers
PATROL_SPEED_RATIO = 10     # Formation drone speed relative to convoy


# ============================================================================
# CORE ALGORITHMS
# ============================================================================

def generate_convoy_trajectory(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    num_waypoints: int = 200
) -> List[Tuple[float, float]]:
    """
    Generate parametric trajectory for convoy movement.

    Uses sinusoidal perturbations to create realistic path with turns
    while maintaining general progression from origin to destination.

    Mathematical Model:
        x(t) = x_0 + (x_f - x_0) * [t + A_x * sin(ω_x * t)]
        y(t) = y_0 + (y_f - y_0) * [t + A_y * sin(ω_y * t)]
        where t ∈ [0, 1]

    Args:
        lon_min: Starting longitude [degrees]
        lon_max: Ending longitude [degrees]
        lat_min: Starting latitude [degrees]
        lat_max: Ending latitude [degrees]
        num_waypoints: Number of discrete waypoints in trajectory

    Returns:
        List of (longitude, latitude) tuples representing convoy path

    Complexity: O(n) where n = num_waypoints
    """
    # Normalized time parameter
    t = np.linspace(0, 1, num_waypoints)

    # Parametric path with sinusoidal perturbations
    # Amplitude = 0.1 (10% of total displacement)
    # Frequencies chosen to create realistic turning behavior
    path_lon = lon_min + (lon_max - lon_min) * (t + 0.1 * np.sin(4 * np.pi * t))
    path_lat = lat_min + (lat_max - lat_min) * (t + 0.1 * np.sin(6 * np.pi * t))

    # Apply boundary constraints with 5% margin
    margin = 0.05
    path_lon = np.clip(
        path_lon,
        lon_min + margin * (lon_max - lon_min),
        lon_max - margin * (lon_max - lon_min)
    )
    path_lat = np.clip(
        path_lat,
        lat_min + margin * (lat_max - lat_min),
        lat_max - margin * (lat_max - lat_min)
    )

    trajectory = list(zip(path_lon, path_lat))

    return trajectory


def generate_formation_patrol_paths(
    leader_trajectory: List[Tuple[float, float]],
    num_agents: int = 20,
    formation_spacing: float = 0.005,
    speed_ratio: int = 10
) -> Dict[int, Dict[str, Any]]:
    """
    Generate patrol paths for formation control drones using oscillatory patrol strategy.

    Strategy: Drones patrol bidirectionally between convoy position and destination,
    maintaining formation structure while maximizing forward area coverage.

    Control Law:
        For agent i at time t:
        - Patrol range: R(t) = [L(t), L_max] where L(t) is leader position
        - Oscillation: p_i(t) = L(t) + A * sin(ω_i * t + φ_i)
        - Formation offset: δ_i = (row_i, col_i) * spacing
        - Final position: x_i(t) = p_i(t) + δ_i

    Args:
        leader_trajectory: Convoy path as list of (lon, lat) tuples
        num_agents: Number of formation drones (default: 20)
        formation_spacing: Inter-agent spacing in formation [degrees]
        speed_ratio: Drone speed relative to convoy (default: 10x)

    Returns:
        Dictionary mapping agent_id to path data containing:
            - type: 'formation'
            - row, col: Position in formation matrix
            - path: List of (lon, lat) waypoints
            - speed_multiplier: Speed ratio
            - initial_spread: Phase offset for patrol oscillation

    Complexity: O(n * m) where n = num_agents, m = len(leader_trajectory)
    """
    agent_paths = {}
    trajectory_length = len(leader_trajectory)

    for agent_id in range(num_agents):
        # Calculate position in formation matrix
        row_idx = agent_id // FORMATION_COLS
        col_idx = agent_id % FORMATION_COLS

        # Formation offset vectors (relative to formation center)
        lon_offset = (col_idx - 1.5) * formation_spacing  # Center on column 2
        lat_offset = -row_idx * formation_spacing * 1.5   # Negative = behind leader

        # Phase offset for patrol oscillation (prevents collision)
        initial_spread = agent_id * 5

        # Generate patrol path using oscillatory control law
        agent_trajectory = []
        for time_step in range(trajectory_length):
            # Compute dynamic patrol range
            patrol_range = trajectory_length - 1 - time_step

            if patrol_range <= 0:
                # Terminal condition: convoy reached destination
                trajectory_idx = trajectory_length - 1
            else:
                # Compute agent position in patrol cycle
                distance_traveled = initial_spread + (time_step * speed_ratio)

                # Bidirectional oscillation with period = 2 * patrol_range
                cycle_length = patrol_range * 2
                position_in_cycle = distance_traveled % cycle_length

                if position_in_cycle <= patrol_range:
                    # Forward sweep toward destination
                    trajectory_idx = time_step + position_in_cycle
                else:
                    # Backward sweep toward convoy
                    trajectory_idx = time_step + (cycle_length - position_in_cycle)

                # Apply boundary constraints
                trajectory_idx = int(np.clip(trajectory_idx, time_step, trajectory_length - 1))

            # Get position on leader trajectory
            leader_pos = leader_trajectory[trajectory_idx]

            # Apply formation offset transformation
            agent_pos = (leader_pos[0] + lon_offset, leader_pos[1] + lat_offset)
            agent_trajectory.append(agent_pos)

        # Store agent path data
        agent_paths[agent_id] = {
            'type': 'formation',
            'row': row_idx,
            'col': col_idx,
            'path': agent_trajectory,
            'speed_multiplier': speed_ratio,
            'initial_spread': initial_spread
        }

    return agent_paths


def generate_perimeter_defense_paths(
    leader_trajectory: List[Tuple[float, float]],
    num_agents: int = 25,
    radius: float = 0.04
) -> Dict[int, Dict[str, Any]]:
    """
    Generate circular perimeter defense paths for protective drones.

    Strategy: Agents uniformly distributed on circle around convoy,
    rotating synchronously to maintain 360° coverage.

    Control Law:
        For agent i at time t:
        - Angular position: θ_i(t) = θ_0,i + ω * t
        - Initial angle: θ_0,i = 2π * i / n
        - Position: x_i(t) = x_leader(t) + r * [cos(θ_i(t)), sin(θ_i(t))]

    Args:
        leader_trajectory: Convoy path as list of (lon, lat) tuples
        num_agents: Number of perimeter drones (default: 25)
        radius: Radius of protective circle [degrees]

    Returns:
        Dictionary mapping agent_id to path data containing:
            - type: 'circle'
            - path: List of (lon, lat) waypoints
            - initial_angle: Starting angular position [radians]

    Complexity: O(n * m) where n = num_agents, m = len(leader_trajectory)
    """
    agent_paths = {}

    # Angular velocity for rotation [rad/timestep]
    angular_velocity = 0.05

    for agent_id in range(num_agents):
        # Uniform angular distribution
        initial_angle = (2 * np.pi * agent_id) / num_agents

        # Generate circular path synchronized with leader
        agent_trajectory = []
        for time_step, leader_pos in enumerate(leader_trajectory):
            # Update angular position with constant angular velocity
            current_angle = initial_angle + (time_step * angular_velocity)

            # Compute Cartesian offset from polar coordinates
            lon_offset = radius * np.cos(current_angle)
            lat_offset = radius * np.sin(current_angle)

            # Apply offset to leader position
            agent_pos = (leader_pos[0] + lon_offset, leader_pos[1] + lat_offset)
            agent_trajectory.append(agent_pos)

        # Store agent path data (offset by 20 to avoid ID collision)
        agent_paths[20 + agent_id] = {
            'type': 'circle',
            'path': agent_trajectory,
            'initial_angle': initial_angle
        }

    return agent_paths


def generate_all_paths() -> Tuple[
    List[Tuple[float, float]],
    Dict[int, Dict[str, Any]],
    Dict[int, Dict[str, Any]]
]:
    """
    Main orchestration function to generate all agent paths.

    Generates:
        1. Convoy trajectory (leader path)
        2. Formation patrol paths (20 drones)
        3. Perimeter defense paths (25 drones)

    Returns:
        Tuple of (convoy_path, formation_drones, perimeter_drones)

    Complexity: O(n * m) where n = total agents (45), m = waypoints (200)
    """
    print("=" * 70)
    print("AUTONOMOUS DRONE PATH PLANNING SYSTEM")
    print("Multi-Agent Convoy Escort with Formation and Perimeter Control")
    print("=" * 70)
    print(f"\nOperational Area:")
    print(f"  Longitude: [{LONGITUDE_MIN:.6f}, {LONGITUDE_MAX:.6f}]")
    print(f"  Latitude:  [{LATITUDE_MIN:.6f}, {LATITUDE_MAX:.6f}]")

    # Phase 1: Generate convoy trajectory
    convoy_path = generate_convoy_trajectory(
        LONGITUDE_MIN, LONGITUDE_MAX,
        LATITUDE_MIN, LATITUDE_MAX,
        num_waypoints=200
    )
    print(f"\n[1/3] Convoy trajectory generated: {len(convoy_path)} waypoints")

    # Phase 2: Generate formation patrol paths
    formation_drones = generate_formation_patrol_paths(
        convoy_path,
        num_agents=20,
        formation_spacing=FORMATION_SPACING,
        speed_ratio=PATROL_SPEED_RATIO
    )
    print(f"[2/3] Formation patrol paths: {len(formation_drones)} agents")
    print(f"      Configuration: {FORMATION_ROWS}×{FORMATION_COLS} matrix")
    print(f"      Speed ratio: {PATROL_SPEED_RATIO}x convoy speed")

    # Phase 3: Generate perimeter defense paths
    perimeter_drones = generate_perimeter_defense_paths(
        convoy_path,
        num_agents=PERIMETER_DRONES,
        radius=PERIMETER_RADIUS
    )
    print(f"[3/3] Perimeter defense paths: {len(perimeter_drones)} agents")
    print(f"      Radius: {PERIMETER_RADIUS:.4f}° (~4.4 km at 36°N)")

    total_agents = len(formation_drones) + len(perimeter_drones)
    print(f"\nTotal UAVs: {total_agents}")
    print("=" * 70)

    return convoy_path, formation_drones, perimeter_drones


def export_paths_to_json(
    convoy_path: List[Tuple[float, float]],
    formation_drones: Dict[int, Dict[str, Any]],
    perimeter_drones: Dict[int, Dict[str, Any]],
    output_filename: str = 'drone_paths.json'
) -> None:
    """
    Serialize all path data to JSON format for interoperability.

    Output Format:
        {
            "metadata": {...},
            "convoy_path": [[lon, lat], ...],
            "formation_drones": {id: {type, row, col, path}, ...},
            "perimeter_drones": {id: {type, path, initial_angle}, ...}
        }

    Args:
        convoy_path: Leader trajectory
        formation_drones: Formation patrol path dictionary
        perimeter_drones: Perimeter defense path dictionary
        output_filename: Output JSON file name

    Side Effects:
        Writes JSON file to disk at /Users/aylenliu/Desktop/hack/{filename}
    """
    # Construct metadata
    metadata = {
        "version": "1.0",
        "algorithm": "Multi-Agent Formation and Perimeter Control",
        "parameters": {
            "formation": {
                "agents": len(formation_drones),
                "rows": FORMATION_ROWS,
                "cols": FORMATION_COLS,
                "spacing": FORMATION_SPACING,
                "speed_ratio": PATROL_SPEED_RATIO
            },
            "perimeter": {
                "agents": len(perimeter_drones),
                "radius": PERIMETER_RADIUS
            },
            "area": {
                "longitude": [LONGITUDE_MIN, LONGITUDE_MAX],
                "latitude": [LATITUDE_MIN, LATITUDE_MAX]
            }
        }
    }

    # Serialize path data
    data = {
        "metadata": metadata,
        "convoy_path": [[float(lon), float(lat)] for lon, lat in convoy_path],
        "formation_drones": {},
        "perimeter_drones": {}
    }

    # Convert formation drone paths
    for agent_id, agent_data in formation_drones.items():
        data['formation_drones'][str(agent_id)] = {
            'type': agent_data['type'],
            'row': agent_data['row'],
            'col': agent_data['col'],
            'path': [[float(lon), float(lat)] for lon, lat in agent_data['path']]
        }

    # Convert perimeter drone paths
    for agent_id, agent_data in perimeter_drones.items():
        data['perimeter_drones'][str(agent_id)] = {
            'type': agent_data['type'],
            'initial_angle': float(agent_data['initial_angle']),
            'path': [[float(lon), float(lat)] for lon, lat in agent_data['path']]
        }

    # Write to file
    output_path = f'/Users/aylenliu/Desktop/hack/{output_filename}'
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nPath data exported to: {output_filename}")
    print(f"File size: {len(json.dumps(data)) / 1024:.2f} KB")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution pipeline for path generation and export.
    """
    # Generate all agent paths
    convoy_path, formation_drones, perimeter_drones = generate_all_paths()

    # Export to JSON format
    export_paths_to_json(convoy_path, formation_drones, perimeter_drones)

  
