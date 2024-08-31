#!/usr/bin/env python
# coding: utf-8

# Script to process and aggregate teacher position log files

import pandas as pd
import numpy as np
import json
import re
import glob

# Function to process a single log file
def process_logfile(f='../data/Position Log/Non Tutor Teaching/2022-05-24-12.48.45EDT.log'):
    with open(f, 'r') as file:
        dd = [json.loads(line.strip())[0] for line in file]
    return pd.json_normalize(dd)

# Function to preprocess position data
def preproc_pos_data(df_pd: pd.DataFrame):
    # Filter for successful records
    df_pd = df_pd[df_pd['success']].copy()
    # Filter for specific tags that were on the teacher (19 and 20)
    df_pd = df_pd[df_pd['tagId'].isin(['200000359', '200000360'])].copy()
    # Map tag IDs to numbers
    df_pd['tag_num'] = df_pd.tagId.map(lambda s: '20' if s == '200000359' else '19')
    return df_pd

# Function to reshape the dataframe for aggregation
def reshape_df(df):
    df = df[['tag_num', 'timestamp', 'data.coordinates.x', 'data.coordinates.y', 'data.position_score']].copy()
    # Round timestamps for grouping.
    # Not exactly like Steven Shou, who used an arbitrary threshold for difference in time stamp (0.5) which is more or less like rounding
    df['timestamp'] = df['timestamp'].map(round)
    # Aggregate by timestamp and calculate weighted average coordinates
    # Steven used only timestamps that have two tag values. Do we want to do the same?
    aggregated_df = df.groupby('timestamp').agg(
        chosen_X=('data.coordinates.x', lambda x: np.average(x, weights=df.loc[x.index, 'data.position_score'])),
        chosen_Y=('data.coordinates.y', lambda y: np.average(y, weights=df.loc[y.index, 'data.position_score']))
    ).reset_index()    
    return aggregated_df.rename(columns={'timestamp': 'time_stamp'})

# Main function to process and reshape a log file, adding period and day IDs
def main_f(f='../data/Position Log/Day 2/Period 5/2022-05-24-14.22.44EDT.log'):
    df = process_logfile(f)
    df = preproc_pos_data(df)
    df = reshape_df(df)
    # Extract periodID and dayID from the file path
    df['periodID'] = re.search('Period (.+?)/', f).group(1)
    df['dayID'] = re.search('Day (.+?)/', f).group(1)
    return df

# Process all log files and concatenate the results
df = pd.concat([main_f(f) for f in glob.glob('../data/Position Log/**/**/*.log')])

# Save the aggregated data to a CSV file
output_file = '../data/2_aggregated_teacher_position.csv'
df.to_csv(output_file, index=False)

# Display the first few rows of the aggregated data
# df.head())

print("...")
print(f"Processed teacher positions have been saved to {output_file}.")
print("...")