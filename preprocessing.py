import pandas as pd
from datetime import datetime
import csv

import os

# Function to convert DOY to month and day
def doy_to_date(year, doy):
    date = datetime(int(year), 1, 1) + pd.Timedelta(days=int(doy) - 1)
    return date.month, date.day

def cleaning_file():
    # Specify the folder path
    folder_path = "../raw_data/rwanda/"  
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            print(filename)
            df = pd.read_csv(folder_path + filename, skiprows=10)

            # Apply the conversion to each row
            df['MO'], df['DY'] = zip(*df.apply(lambda row: doy_to_date(row['YEAR'], row['DOY']), axis=1))

            # Select and reorder the columns
            output_df = df[['YEAR', 'MO', 'DY', 'PRECTOTCORR', 'TS']]

            # Save to a new CSV file
            output_df.to_csv('dataset/rwanda/'+ filename + '_clean.csv', sep=";", index=False)

    print("Conversion complete.")



def extract_lat_lon(input_folder, output_file, countryname):
    # Write to output CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        folder_path = input_folder+countryname+"/"  
        for filename in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, filename)):
                print(filename)   
                # Read the input file to find latitude and longitude
                with open(folder_path+filename , 'r') as f:
                    for line in f:
                        if 'latitude' in line:
                            # Extract latitude and longitude values
                            parts = line.split()
                            lat = float(parts[2])
                            lon = float(parts[4])
                            break

                    writer.writerow([countryname, filename[:-4], lat, lon])
    
                    print(filename[:-4])

# Example usage:
#input_csv = 'afar.csv'
#output_csv = 'location_data.csv'
#country = 'Ethiopia'  # Replace with actual country name
#site = 'Afar'         # Replace with actual site name

#extract_lat_lon(input_csv, output_csv, country, site)

cleaning_file()

#input_folder = "../raw_data/"
#output_file = "dataset/location_data.csv"
#countryname = "ethiopia"

#extract_lat_lon(input_folder, output_file, countryname)