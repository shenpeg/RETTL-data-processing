#!/usr/bin/env python
# coding: utf-8

# This is a boilerplate code ONLY
# Script to join student position data with seating chart data

import pandas as pd

# Load the seating chart data
seating_chart_file = '../data/seating_chart_coords.csv'
df_seats = pd.read_csv(seating_chart_file)

# Clean the 'object' column in seating chart data by removing 'seat' from the values
df_seats['object'] = df_seats['object'].map(lambda s: s.replace('seat', ''))

# Load the student position data
student_position_file = '../data/0_complete_student_positions.csv'
df_students = pd.read_csv(student_position_file)[['dayID', 'periodID', 'seatNum', 'actual_user_id']]
df_students['seatNum'] = df_students['seatNum'].map(str)

# Do the same for datashop crosswalk from cindy t to join anon_user_id...

# Merge the student position data with seating chart data on seat number
merged_df = df_students.merge(df_seats, how='outer', left_on='seatNum', right_on='object')

# Save the merged data to a new CSV file
output_file = '../data/1_merged_student_position.csv'
merged_df.to_csv(output_file, index=False)

print("...")
print(f"Merged student positions have been saved to {output_file}.")
print("...")