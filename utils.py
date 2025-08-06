import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler

import pandas as pd

def get_coordinates(country_name, city_name):
    # Read the CSV file into a DataFrame
    csv_file = "dataset_drought/cities_coordinates.csv"
    df = pd.read_csv(csv_file)
    
    # Normalize strings by stripping and lowercasing
    normalized_city = city_name.strip().lower()
    normalized_country = country_name.strip().lower()
    
    df['norm_city'] = df['city'].str.strip().str.lower()
    df['norm_country'] = df['country'].str.strip().str.lower()
    
    # Filter rows
    filtered = df.loc[
        (df['norm_city'] == normalized_city) & 
        (df['norm_country'] == normalized_country)
    ].copy()

    if len(filtered) == 0:
        print(f"No match found for: City='{normalized_city}', Country='{normalized_country}'")
        print("Available cities in this country:", 
              df[df['norm_country'] == normalized_country]['city'].unique())
        return None
    else:
        longitude = filtered.iloc[0]['longitude']
        latitude = filtered.iloc[0]['latitude']
        return latitude, longitude

import requests
from datetime import datetime
import pandas as pd
from time import sleep


def get_monthly_weather(country_name,city_name, year, month):
    latitude , longitude = get_coordinates(country_name, city_name)
    """
    Fetch hourly TS (skin temp) and PRECTOTCORR (precipitation) in one API call.
    Returns:
        - Monthly average TS (°C)
        - Monthly total precipitation (mm)
    """
    # Set date range for the month
    start_date = f"{year}{month:02d}01"
    end_date = (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)).strftime("%Y%m%d")
    
    # NASA POWER API request
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "TS,PRECTOTCORR",  # Request both variables
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code != 200:
        raise ValueError(f"API Error: {data.get('message', 'Unknown error')}")
    
    # Extract hourly data
    ts_values = [
        float(val) 
        for val in data["properties"]["parameter"]["TS"].values() 
        if val != -999  # Skip missing TS
    ]
    precip_values = [
        float(val) 
        for val in data["properties"]["parameter"]["PRECTOTCORR"].values() 
        if val != -999  # Skip missing precipitation
    ]
    
    # Compute monthly stats
    avg_ts = sum(ts_values) / len(ts_values)  # TS in °C (already in °C)
    total_precip = sum(precip_values)  # PRECTOTCORR in mm
    
    return total_precip, avg_ts




def create_model(x_train, y_train, x_test, y_test): 
    models = []
    models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
    models.append(('LR', LogisticRegression(random_state=42)))
    models.append(('SVC', SVC(random_state=42)))
    models.append(('DT', DecisionTreeClassifier(random_state=42)))
    models.append(('ETC', ExtraTreesClassifier(random_state=42)))
    models.append(('RF', RandomForestClassifier(random_state=42)))
    models.append(('XGB', XGBClassifier(n_estimators=1000, learning_rate=0.01,random_state=42)))
    names = []
    accuracy = []
    precision = []
    recall = []
    specificity = []
    
    for name, model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        #precision.append(precision_score(y_test, y_pred, average='weighted'))
        #recall.append(recall_score(y_test, y_pred, average='weighted'))
        #specificity.append(recall_score(y_test, y_pred, average='weighted', pos_label=0))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        specificity.append(recall_score(y_test, y_pred, pos_label=0))
        names.append(name)

    models_perf = pd.DataFrame({'Ids': [0,1,2,3,4,5,6], 'Name': names, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'Specificity': specificity})
    models_perf_sorted = models_perf.sort_values(by=['Recall', 'Precision', 'Accuracy', 'Specificity'], ascending=[False, False, False, False])

    # Extract the best model and its metrics
    best_model = models_perf_sorted.iloc[0]
    model = models[int(best_model['Ids'])][1]

    return model, best_model['Name'], best_model['Accuracy'], best_model['Precision'], best_model['Recall'], best_model['Specificity']


def create_dataset(country, sitename, spei_threshold, diff_month):

    path="dataset_drought/"+country+"/"
    file1 = sitename+"_clean.csv"
    file2 = sitename+"_spei.csv"
    # read the data df
    df = pd.read_csv(path+file1,sep=";")
    # rename columns for dates
    df = df.rename(columns={"YEAR":"year","MO":"month","DY":"day"})
    # create the date column 
    df["dates"] = pd.to_datetime(df[["year","month","day"]])
    
    # read the spei dataset
    df_spei = pd.read_csv(path+file2,sep=";")
    
    # convert the dates column to the correct type
    df['dates'] = pd.to_datetime(df['dates'])
    df_spei['dates'] = pd.to_datetime(df_spei['dates'])
    
    # realize the left join on dates column
    res = pd.merge(df, df_spei, on='dates', how="left")
    features = ["dates","spei01","TS","PRECTOTCORR"]
    res2 = aggregate_data(res, features)

    # apply the class format to all the spei columns of the dataframes
    res2["class"] = res2["spei"].apply(lambda x: from_spei_to_class(x, spei_threshold))
    # add the month columns
    res2["month"] = res2["dates"].dt.month
    # Convert month to 4-bit binary string (e.g., 5 → '0101')
    res2['month_binary'] = res2['month'].apply(lambda x: bin(x)[2:].zfill(4))

    # Split binary string into 4 columns
    for i in range(4):
        res2[f'pmonth_bit_{i}'] = res2['month_binary'].str[i].astype(int)
    for i in range(4):
        res2[f'month_bit_{i}'] = res2['month_binary'].str[i].astype(int)
    
    # Drop the intermediate month column if needed
    res2 = res2.drop('month', axis=1)
    #res2 = res2.drop('prev_month', axis=1)
    res2 = res2.drop('month_binary', axis=1)

    # select only the good features
    final_features = ["temperature","precipitations", 
                  "pmonth_bit_0", "pmonth_bit_1", "pmonth_bit_2", "pmonth_bit_3",
                  "month_bit_0", "month_bit_1", "month_bit_2", "month_bit_3", "class"]
    res3 = res2[final_features]
     

    res3[['temperature', 'precipitations', "pmonth_bit_0", "pmonth_bit_1", "pmonth_bit_2", "pmonth_bit_3"]] = res3[['temperature', 'precipitations', "pmonth_bit_0", "pmonth_bit_1", "pmonth_bit_2", "pmonth_bit_3"]].shift(diff_month)
    
    for i in range(diff_month):
        res3 = res3.drop([i, len(res3) - 1])


    # Instanciate an oversampler
    oversampler = RandomOverSampler(random_state=42)
    # Get X and y
    x, y = res3.drop("class", axis=1), res3["class"]
    
    # Oversample the data
    x_p_o, y_p_o = oversampler.fit_resample(x, y)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_p_o, y_p_o, test_size=0.3, random_state=1)
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    return x_train,x_test,y_train,y_test

# transform spei scores into classes
#def from_spei_to_class(spei):
#    if(spei<-1.5):
#        return 2 #  Extreme drought
#    elif(spei>=-1.50 and spei<-0.50):
#        return 1 # Drought
#    else:
#        return 0 # No Drought

# transform spei scores into classes
def from_spei_to_class(spei, spei_threshold):
    if(spei<-spei_threshold):
        return 1 # Drought
    else:
        return 0 # No Drought

# define the function which agregates data
def aggregate_data(data, features,col="spei01"):
    
    # find the indexes for non null values of the column col
    non_null_indexes = data.loc[data[col].notnull()].index
    
    # define the agg lists
    mean_aggregates = []
    median_aggregates  = []
    sum_aggregates = []
    
    # we go through the non-zero indices to work with
    for index in non_null_indexes:
        subset = data[features].iloc[:index+1]
        
        # avg agg
        mean_agg = subset.drop(columns=[col,"dates"],axis=1).mean()
        # median agg
        median_agg = subset.drop(columns=[col,"dates"],axis=1).median()
        # sum agg
        sum_agg = subset.drop(columns=[col,"dates"],axis=1).sum()
        
        # add each aggregate in the corresponding list
        mean_aggregates.append(mean_agg)
        median_aggregates.append(median_agg)
        sum_aggregates.append(sum_agg)
    
    
    # set the date for mean_df
    mean_df = pd.DataFrame(mean_aggregates)
    dates = data.loc[non_null_indexes,"dates"]
    spei = data.loc[non_null_indexes,"spei01"]
    dates.index = mean_df.index
    spei.index = mean_df.index
    mean_df["dates"] = dates
    mean_df["spei"] = spei
    
    # set the date for med_df
    med_df = pd.DataFrame(median_aggregates)
    med_df["dates"] = dates
    med_df["spei"] = spei
    
    # set the date for sum_df
    sum_df = pd.DataFrame(sum_aggregates)
    sum_df["dates"] = dates
    sum_df["spei"] = spei
    
    # build the final dataframe to return with all the agregates
    final_df = pd.DataFrame(columns=["temperature","precipitations","dates","spei"])
    final_df["temperature"] = med_df["TS"]
    final_df["precipitations"] = sum_df["PRECTOTCORR"]
    final_df["dates"] = dates
    final_df["spei"] = spei
    
    # return the unified dataframe
    return final_df

def month_to_digit(month):
    months_letters= {
        "january":1,
        "february":2,
        "march":3,
        "april":4,
        "may":5,
        "june":6,
        "july":7,
        "august":8,
        "september":9,
        "october":10,
        "november":11,
        "december":12
    }
    
    if month in months_letters:
        return months_letters[month]
    else:
        return "Invalid Month"
  