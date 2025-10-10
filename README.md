# AutoGators: Autonomous Convoy Routing System

**Winner — “Most Innovative Solution” ($10,000)**  
Awarded at the **AWS × Vanderbilt Mission Autonomy Hackathon (Oct 2025)**.  

## Overview

AutoGators is a system designed to enable autonomous path planning for ground convoys in complex, resource-constrained environments where connectivity to a centralized command center may be limited. The system integrates aerial reconnaissance, threat detection, and predictive routing to safely guide convoys through dynamic battlefields or hazard zones.

The core idea is to allow convoys to adapt to changing conditions in real time by combining drone-based threat detection with a tactical routing engine.

---

## System Design

### Architecture

1. **Aerial Reconnaissance (Drones)**
   - A swarm of drones performs continuous surveillance of the area.
   - Drones detect both static threats (fires, obstacles, landmines) and dynamic threats (mobile adversaries).
   - Threat data is processed and converted into map polygons for use by the routing system.

2. **Threat Analysis**
   - Detected threats are classified into categories.
   - Threat polygons are used to remove or penalize edges in the navigation graph to avoid unsafe paths.

3. **Routing Engine**
   - Graph-based representation of terrain and roads.
   - Computes alternate routes using shortest-path algorithms.
   - Introduces a penalty system for hostile or risky areas to prioritize safer paths.
   - Supports multiple alternate routes to reduce single points of failure.
   - Predictive component anticipates high-risk zones and incorporates them in route selection.

4. **User Interface**
   - Interactive GUI displays the tactical map, start and end points, threats, hostile zones, and calculated routes.
   - Allows operators to select routes, update start/end points, and visualize hostile exposure and path diversity.
   - Built using Streamlit and Folium for real-time interactivity.

5. **Computer Vision Module**
   - Processes aerial imagery to detect threats.
   - Uses trained classifiers with optional logit bias tuning.
   - Supports augmentation strategies like Mixup and CutMix for robust training.

6. **Integration with Generative Models**
   - Optional AI-assisted scenario simulation (e.g., visualizing flooding or damage) using VLM APIs.
   - Supports image variation and environment perturbation to predict potential hazards dynamically.

---

## Key Features

- **Dynamic Threat Awareness:** Real-time updates from drones feed directly into the routing engine.
- **Alternate Path Planning:** Multiple routes calculated with penalties for risk exposure and edge diversity.
- **Hostile Area Penalties:** Graph edges intersecting risky zones are penalized to reduce probability of convoy exposure.
- **Predictive Routing:** Avoids single points of failure by planning around critical chokepoints.
- **Interactive Map:** Operators can visualize routes, threats, and hostile areas with color-coded feedback.
- **Machine Learning Integration:** Classifiers detect and categorize hazards, enhancing threat awareness.

---

## Design Principles

- **Modularity:** Drone input, threat analysis, and routing engine operate as independent modules for flexibility.
- **Resilience:** System continues functioning under partial information or intermittent connectivity.
- **Adaptivity:** Convoys can reroute in real-time based on new threats.
- **Transparency:** GUI provides clear visual feedback on route safety, distance, and risk.

---

## Future Extensions

- Integration with swarm coordination algorithms for drones.
- Predictive modeling of adversary movement using ML.
- Multi-convoy coordination for simultaneous resupply missions.
- Simulation of dynamic environmental hazards for training and testing.

---

## Usage

1. Launch the Streamlit interface for route calculation.
2. Load aerial imagery or live threat data.
3. Set start and end points for the convoy.
4. Calculate alternate paths and review route analysis.
5. Select optimal route based on risk exposure and travel time.

