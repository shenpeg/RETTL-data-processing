#!/usr/bin/env python3
"""
This script processes teacher position data to detect and analyze classroom visit events.

Files:
    F_POS (str): Path to the teacher position data CSV file.
    F_CHART (str): Path to the seating chart data CSV file.

Parameters (set based on conditional maximization in EDM23 paper with visits used in AIED23 and LAK24 papers):
    DURATION (int): Duration threshold for stop detection. 21
    RADIUS (int): Radius threshold for stop detection. 600
    RANGE (int): Range threshold for determining proximity to students once stop is established. 700

Procedure:
1. Load teacher position data and seating chart data.
2. Use the visit detection function to infer visit events based on the specified thresholds.
3. Filter inferred visit events.
4. Group and aggregate the inferred visit events by stop index to obtain a table of unique visits.
5. Save the processed visit events to a CSV file.

Output:
    A CSV file ('visits.csv') containing unique visits, including timestamps, seats visited, centroids, period IDs, and day IDs.

References:
    Shou, T., Borchers, C., Karumbaiah, S., & Aleven, V. (2023). Optimizing Parameters for Accurate Position Data Mining in Diverse Classrooms Layouts. In Proceedings of the 16th International Conference on Educational Data Mining (EDM). Bengaluru, India.
"""

import ast
import re
import itertools
import evalfuns as ef

# File paths
F_POS = '../data/2_aggregated_teacher_position.csv'
#F_POS = '../k-12-data/teacher_position_sprint1_shou.csv'
F_CHART = '../data/seating_chart_coords.csv'

# Parameters for visit detection
DURATION = 21
RADIUS = 600
RANGE = 700

# Detect visit events based on teacher position data
inferred_events = ef.visit_detection_based_on_position_data_routine(
    f_teacher=F_POS, f_chart=F_CHART,
    duration=DURATION, radius=RADIUS, rng=RANGE
)

# Filter inferred visit events
inferred_visit_events = inferred_events[inferred_events.possibleSubjects.map(lambda d: len(str(d)) > 3)].copy()
inferred_visit_events['seats'] = inferred_visit_events['possibleSubjects'].map(lambda s: re.findall(r'seat[0-9]+', str(s)))

# Group and aggregate the inferred visit events by stop index to obtain unique visits
res = inferred_visit_events.groupby('stop_index').agg(
    min_timestamp=('timestamp', 'min'),
    max_timestamp=('timestamp', 'max'),
    seats=('seats', lambda x: list(set(itertools.chain.from_iterable(x)))),
    centroid=('centroid', lambda x: list(x.unique())),
    periodID=('periodID', lambda x: list(x.unique())),
    dayID=('dayID', lambda x: list(x.unique()))
).reset_index()

# Save the processed visit events to a CSV file
output_file = '../data/3_visits.csv'
res.to_csv(output_file, index=False)

# Print completion messages
print("...")
print(f"Visits have been detected and saved to {output_file}.")
print("...")