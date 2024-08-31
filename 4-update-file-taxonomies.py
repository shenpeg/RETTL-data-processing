#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
from datetime import datetime
import os

# File path to the crosswalk data (bell schedule sessions)
F_CROSSWALK = '../data/bell_schedule_sessions.csv'

def unix_to_utc(unix_timestamp):
    """
    Converts a Unix timestamp to a standard UTC format string.

    Parameters:
    unix_timestamp (int): Unix timestamp in seconds.

    Returns:
    str: UTC formatted date string.
    """
    # Convert the Unix timestamp to a datetime object
    dt = datetime.utcfromtimestamp(unix_timestamp)
    # Format the datetime object to a string in UTC format
    utc_format = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    return utc_format

# Create a new directory for processed files to put the files in
processed_dir = '../data/Processed Files'
os.makedirs(processed_dir, exist_ok=True)

# ## UPDATED STUDENT POSITION

# Load the crosswalk data
df_crosswalk = pd.read_csv(F_CROSSWALK)

# File path to the merged student position data
F_POS = '../data/1_merged_student_position.csv'
df_student = pd.read_csv(F_POS)

#df_student.drop_duplicates(subset=['periodID', 'dayID'])[['periodID', 'dayID']].sort_values(by=['periodID', 'dayID'])

# Merge student position data with crosswalk data
df_student_new = df_student.merge(df_crosswalk, how='left', on=['periodID', 'dayID'])

#df_student.shape

# Select and rename columns for output
df_student_new_out = df_student_new\
    [['School', 'Class', 'Short name', 'actual_user_id', 'X', 'Y']]\
    .rename(columns={'actual_user_id': 'Student name'})

# Save the student position files for each class
for cl in set(df_student_new_out.Class):
    tmp = df_student_new_out[df_student_new_out['Class']==cl].copy()
    tmp.to_csv(f'{processed_dir}/student-position-{cl}-amended.csv', index=False)

print("...")
print(f"Student position files have been post-processed and saved to {processed_dir}.")
print("...")

# ## UPDATED TEACHER POSITION

# File path to the aggregated teacher position data
F_POS = '../data/2_aggregated_teacher_position.csv'
df_teacher = pd.read_csv(F_POS)

# df_teacher['X_avg'] = (df_teacher['tag19_X']*df_teacher['tag19_score'] + df_teacher['tag20_X']*df_teacher['tag20_score']) / (df_teacher['tag19_score']+df_teacher['tag20_score'])
# df_teacher['Y_avg'] = (df_teacher['tag19_Y']*df_teacher['tag19_score'] + df_teacher['tag20_Y']*df_teacher['tag20_score']) / (df_teacher['tag19_score']+df_teacher['tag20_score'])

# Merge teacher position data with crosswalk data and rename output columns
df_teacher_out = df_teacher\
    .merge(df_crosswalk, how='left', on=['periodID', 'dayID'])\
    .rename(columns={'chosen_X': 'X', 'chosen_Y': 'Y', 'time_stamp': 'Time'})\
    [['School', 'Class', 'Short name', 'Time', 'X', 'Y']]

# Convert timestamps to UTC format
df_teacher_out['Time'] = df_teacher_out['Time'].map(unix_to_utc)

# Save the teacher position files for each class
for cl in set(df_teacher_out.Class):
    tmp = df_teacher_out[df_teacher_out['Class']==cl].copy()
    tmp.to_csv(f'{processed_dir}/teacher-position-{cl}-amended.csv', index=False)

print(f"Teacher position files have been post-processed and saved in {processed_dir}.")
print("...")

# ## UPDATED STUDENT VISIT DISTILLATION

import ast

# Load visit data
df_visits = pd.read_csv('../data/3_visits.csv')

# Extract X and Y coordinates from centroid data
df_visits['X'] = df_visits.centroid.map(lambda s: s.split(',')[0].split('(')[-1])
df_visits['Y'] = df_visits.centroid.map(lambda s: s.split(',')[-1].split(')')[0])

# Extract periodID and dayID from string representation
df_visits['periodID'] = df_visits.periodID.map(lambda s: s.split(']')[0].split('[')[-1])
df_visits['dayID'] = df_visits.dayID.map(lambda s: s.split(']')[0].split('[')[-1])

# Create a lookup dictionary for student names based on period-day-seat
df_student['period-day-seat'] = df_student['periodID'].map(str) + '-' + df_student['dayID'].map(str) + '-seat' + df_student['seatNum'].map(str)
d_lookup = {row['period-day-seat']: row['actual_user_id'] for _, row in df_student[['period-day-seat', 'actual_user_id']].iterrows()}

# Map visit events to student names
ans = []
for _, row in df_visits.iterrows():
    pe_day = str(row['periodID']) + '-' + str(row['dayID']) + '-'
    ans.append([d_lookup.get(pe_day + x) for x in ast.literal_eval(row['seats']) if d_lookup.get(pe_day + x) is not None])
df_visits['Student name'] = ans

# Filter out visit events with no student names (empty seats)
df_visits = df_visits[df_visits['Student name'].map(lambda l: len(str(l)) > 2)].copy()

# Convert periodID and dayID to integers
df_visits['periodID'] = df_visits['periodID'].map(int)
df_visits['dayID'] = df_visits['dayID'].map(int)

# Merge visit data with crosswalk data and rename columns for output
df_visits_out = df_visits\
    .merge(df_crosswalk, how='left', on=['periodID', 'dayID'])\
    .rename(columns={'min_timestamp': 'Time min', 'max_timestamp': 'Time max'})\
    [['School', 'Class', 'Short name', 'Time min', 'Time max', 'X', 'Y', 'Student name']]

# Clean up column names and format
df_visits_out.rename(columns={'Student name': 'Student names'}, inplace=True)
df_visits_out.rename(columns={'Time min': 'Start time'}, inplace=True)
df_visits_out.rename(columns={'Time max': 'End time'}, inplace=True)
df_visits_out['Student names'] = df_visits_out['Student names'].map(lambda l: json.dumps(l))
df_visits_out['Start time'] = df_visits_out['Start time'].map(unix_to_utc)
df_visits_out['End time'] = df_visits_out['End time'].map(unix_to_utc)

# Save the visit files for each class
for cl in set(df_visits_out.Class):
    tmp = df_visits_out[df_visits_out['Class']==cl].copy()
    tmp.to_csv(f'{processed_dir}/visits-{cl}-amended.csv', index=False)

print(f"Visit files have been post-processed and saved in {processed_dir}.")
print("...")

print(f"All files have been post-processed and are ready to be uploaded to the server!")
print("...")