import pandas as pd

# Load the input files
student_positions_file = '../data/student_seating_positions.csv'
seating_chart_file = '../data/seating_chart_coords.csv'
bell_schedule_file = '../data/bell_schedule_sessions.csv'

df_student_positions = pd.read_csv(student_positions_file)
df_seating_chart = pd.read_csv(seating_chart_file)
df_bell_schedule = pd.read_csv(bell_schedule_file)

# Get the unique dayIDs from the student positions file
valid_dayIDs = df_student_positions['dayID'].unique()

# Filter the bell schedule to include only valid dayIDs
df_bell_schedule = df_bell_schedule[df_bell_schedule['dayID'].isin(valid_dayIDs)]

# Create a list to hold all rows, including the empty seats
all_rows = []

# Iterate over each unique day and period from the bell schedule
for _, row in df_bell_schedule.iterrows():
    dayID = row['dayID']
    periodID = row['periodID']
    
    # Get the current period's student positions
    current_positions = df_student_positions[(df_student_positions['dayID'] == dayID) & 
                                             (df_student_positions['periodID'] == periodID)]
    
    # Check if there are students for the current day
    if current_positions.empty:
        continue
    
    # Create a set of occupied seats for the current period
    occupied_seats = set(current_positions['seatNum'])
    
    # Add existing student positions to all_rows
    all_rows.extend(current_positions.to_dict('records'))
    
    # Iterate over each seat in the seating chart
    for _, seat_row in df_seating_chart.iterrows():
        seatNum = int(seat_row['object'].replace('seat', ''))
        X = seat_row['X']
        Y = seat_row['Y']
        
        # If the seat is not occupied, add an empty seat entry
        if seatNum not in occupied_seats:
            empty_seat_entry = {
                'dayID': dayID,
                'periodID': periodID,
                'seatNum': seatNum,
                'actual_user_id': 'NOSTUDENT',
                'X': X,
                'Y': Y,
                'anon_user_id': 'NA'
            }
            all_rows.append(empty_seat_entry)

# Create a DataFrame from all_rows
df_all_positions = pd.DataFrame(all_rows)

# Save the DataFrame to a new CSV file
output_file = '../data/0_complete_student_positions.csv'
df_all_positions.to_csv(output_file, index=False)

print("...")
print(f"Complete student positions have been saved to {output_file}.")
print("...")