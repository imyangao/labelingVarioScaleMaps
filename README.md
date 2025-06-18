# Labeling Vario-Scale Maps

This repository provides necessary components for generating, processing, and evaluating label anchors for vario-scale maps. It includes the computation of label anchor points for geometries, label trace IDs for label line trajectories, evaluation metrics, and export JSON files for visualization labels on the web.

## Overview

We compute **label anchor points** for features like roads, water, and buildings across multiple generalization steps. It uses geometric skeletons or centorid/pole of inaccesbility depending on the feature type. It traces label anchors over steps for consistency and interpolation, storing the results in PostgreSQL/PostGIS.

---

## File Descriptions

### `labeling_core/anchors.py`

Computes label anchor points and rotation angles for a given geometry:
- For roads/water: uses skeleton lines and midpoints of longest segments.
- For buildings: uses centroid or polylabel plus orientation of the longest edge of the minimum bounding rectangle.

### `labeling_core/db.py`

Handles PostgreSQL/PostGIS database connection setup. Also defines constants for identifying feature types (road, water, buildings) based on `feature_class` codes.

### `labeling_core/skeleton.py`

Core utility functions for:
- Extracting polygon boundary segments.
- Generating skeleton lines using the [Grassfire algorithm](https://github.com/bmmeijers/grassfire) by [Martijn Meijers](https://github.com/bmmeijers).
- Building graphs from skeleton lines.
- Finding main paths between important junctions.
- Processing all layers in a GeoPackage to produce skeletons, label anchors, and outputs(optional).

### `labeling_core/traces.py`

Links anchors across multiple steps into **label_trace_IDs** using spatial proximity.

### `label4event.py`

Main pipeline for event-based method to:
- Reconstruct polygon geometry per event.
- Compute anchors per face using `anchors.py`.
- Store results in `label_anchors` table.
- Assign label_trace_IDs to anchors using `traces.py`.

### `label4slice_multi.py`

Main pipeline for slice-based method to:
- Proecess either from GeoPackage files or directly from PostGIS.
- Store anchor results in `label_anchors_from_slices`.
- Merge labels across slices and assigns label_trace_IDs.

### `event_statistics.py`

Performs simple statistics:
- Counts how many key steps each face has.
- Plots a bar chart showing the frequency of top 50 anchor updates per face.

### `evaluation_anchor_transitions.py`

Evaluates label trace smoothness across steps:
- Interpolates anchor points and angles between steps.
- Computes jump distances between consecutive steps.
- Outputs statistics (mean, p99, maximum) of jump distances per method (event-based vs slice-based).

### `labels2JSON.py`

Exports anchor points from the database table to a JSON file:
- Each entry includes `label_id`, `face_id`, geometry as GeoJSON, angle, and label trace ID.
- Used for downstream consumption in rendering. More information could be checked here: https://github.com/imyangao/addLabels2Map.

### `scalestep.py`

Provides the mapping between generalization step values and map scale denominators, this code is a work from [Martijn Meijers](https://github.com/bmmeijers).

---
