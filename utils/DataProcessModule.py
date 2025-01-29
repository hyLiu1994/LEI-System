import requests, os
import matplotlib.pyplot as plt
from datetime import datetime
import zipfile
import pandas as pd
from tqdm import tqdm
from utils.RTree import format_mbr


time_col_name = "# Timestamp"
Lat_col_name = "Latitude"
Lon_col_name = "Longitude"
time_formulation = "%d/%m/%Y %H:%M:%S"

def unzip_file(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"File '{zip_path}' has been unzipped.")
    except zipfile.BadZipFile:
        print(f"Error: The file '{zip_path}' is not a valid ZIP file.")
    except Exception as e:
        print(f"Error unzipping the file '{zip_path}': {e}")

def get_PositionID_based(position, grid_size=0.01):
    total_lat_grids = int(180 / grid_size)
    total_lon_grids = int(360 / grid_size)
    latitude, longitude = position[0], position[1]

    lat_grid = int((latitude + 90) / grid_size)
    lon_grid = int((longitude + 180) / grid_size)

    lat_grid = min(max(lat_grid, 0), total_lat_grids - 1)
    lon_grid = min(max(lon_grid, 0), total_lon_grids - 1)

    return lat_grid * total_lon_grids + lon_grid    

def download_aisdk_dataset(file_name, day):
    # Step 1: Download Raw Dataset
    if ("aisdk" in file_name):
        download_url = "http://web.ais.dk/aisdata/"
    else:
        download_url = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/"

    url = download_url + file_name + ".zip"
    file_dir = "./Data/" + file_name + "/"
    file_name = file_name + ".zip"
    if not os.path.exists(file_dir + file_name):
        try:
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_length = int(response.headers.get('content-length'))
            with open(file_dir + file_name, 'wb') as file, tqdm(
                desc=file_dir + file_name,
                total=total_length,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = file.write(chunk)
                    bar.update(size)
            print(f"ZIP file downloaded successfully as {file_dir + file_name}")

            unzip_file(file_dir + file_name, file_dir)

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
        except Exception as e:
            print(f"Error downloading the ZIP file: {e}")
    else:
        print(f"File '{file_dir + file_name}' already exists. No download needed.")

    # Step 2: Set base information about raw dataset
    global time_col_name, Lat_col_name, Lon_col_name, time_formulation
    if ("aisdk" in file_name):
        time_col_name = "# Timestamp"
        Lat_col_name = "Latitude"
        Lon_col_name = "Longitude"
        time_formulation = "%d/%m/%Y %H:%M:%S"
    else:
        time_col_name = "BaseDateTime"
        Lat_col_name = "LAT"
        Lon_col_name = "LON"
        time_formulation = "%Y-%m-%dT%H:%M:%S"
    
    # Step 3: Return csv_file_name
    if (file_name.count('-') == 2 or file_name.count('_') == 2):
        csv_file_name = file_name[:10].replace('-', '_') + file_name[11:-4] + day + ".csv"
    else:
        csv_file_name = file_name[:-4] + '.csv'

    return csv_file_name

def filter_invalid_coordinates(data, ship_id, times_interval, ship_data):
    def is_valid_coordinate(lat, lon):
        return -90 <= lat <= 90 and -180 <= lon <= 180
    seq_id = 0
    filtered_data = {
        'timestamps': [],
        'latitudes': [],
        'longitudes': []
    }
    pre = -1
    for timestamp, lat, lon in zip(data['timestamps'], data['latitudes'], data['longitudes']):
        timestamp = datetime.strptime(timestamp, time_formulation).timestamp()
        if is_valid_coordinate(lat, lon):
            if (pre == -1 or timestamp - pre <= times_interval):
                filtered_data['timestamps'].append(timestamp)
                filtered_data['latitudes'].append(lat)
                filtered_data['longitudes'].append(lon)
                pre = timestamp
            else:
                # print(str(ship_id) + '_' + str(seq_id))
                if (len(filtered_data['timestamps']) >= 5):
                    ship_data[str(ship_id) + '_' + str(seq_id)] = filtered_data
                    seq_id += 1
                filtered_data = {
                    'timestamps': [],
                    'latitudes': [],
                    'longitudes': []
                }
                pre = -1
    if (pre != -1):
        if (len(filtered_data['timestamps']) > 1):
            ship_data[str(ship_id) + '_' + str(seq_id)] = filtered_data
            seq_id += 1

def analysis_frequency_time_interval(df, file_name, day, ratio = 2.0/3, file_dir = "./"):
    df['Second'] = pd.to_datetime(df[time_col_name], format=time_formulation)
    df.sort_values(by=['MMSI', 'Second'], inplace=True)
    df['Time_diff'] = df.groupby('MMSI')['Second'].diff().dt.total_seconds()
    df['Time_diff'].fillna(0, inplace=True)
    df['Time_diff'] = df['Time_diff'].astype(int)
    time_diff_freq = df['Time_diff'].value_counts().reset_index()
    time_diff_freq.columns = ['Time_diff', 'Frequency']
    plt.figure(figsize=(10, 6))
    plt.bar(time_diff_freq['Time_diff'], time_diff_freq['Frequency'], width=100)  # 调整宽度以适应您的数据
    plt.xlim(0, 1000)
    plt.xlabel('Time Interval')
    plt.ylabel('Frequency (log scale)') 
    plt.yscale('log') 
    plt.title(str(file_name) + str(day))
    plt.savefig(file_dir + 'Figure/frequency_distribution' + str(file_name) + '_' + str(day) + '.png')

    time_diff_freq.sort_values('Time_diff', inplace=True)
    total_frequency = time_diff_freq['Frequency'].sum()
    time_diff_freq['Cumulative_Percentage'] = time_diff_freq['Frequency'].cumsum() / total_frequency * 100

    plt.figure(figsize=(10, 6))
    plt.plot(time_diff_freq['Time_diff'], time_diff_freq['Cumulative_Percentage'], marker='o', linestyle='-')
    plt.xlim(0, 1000)
    plt.xlabel('Time Interval')
    plt.ylabel('Percentage of Data <= Interval (%)')
    plt.title('Cumulative Percentage of Data <= Each Time Interval')
    plt.grid(True)
    plt.savefig(file_dir + 'Figure/cumulative_percentage_distribution_' + str(file_name) + '_' + str(day) + '.png')

    total_data_points = len(df)
    time_diff_freq_sorted = time_diff_freq.sort_values(by='Time_diff')
    time_diff_freq_sorted['Cumulative_Frequency'] = time_diff_freq_sorted['Frequency'].cumsum()

    threshold_frequency = ratio * total_data_points
    selected_interval = time_diff_freq_sorted.loc[time_diff_freq_sorted['Cumulative_Frequency'] >= threshold_frequency, 'Time_diff'].iloc[0]
    print("Selected Interval:", selected_interval, "seconds,  ", "ratio: ", ratio * 100, "%.")
    return selected_interval

def load_aisdk_dataset(file_name, day, connection_ratio=0.99, datascalability=-1, org_file_dir = "./", is_download=True):
    # Step 1: Load Raw Dataset
    if is_download:
        csv_file_name = download_aisdk_dataset(file_name, day)

        # Step 2: filter Raw Dataset
        file_dir = org_file_dir + "Data/" + file_name + "/"
        csv_file = file_dir + csv_file_name

    else:
        csv_file_list = get_csv_files(org_file_dir)
        csv_file = csv_file_list[0]


    df = pd.read_csv(csv_file)
    if (datascalability != -1):
        df = df.head(datascalability)

    selected_interval = analysis_frequency_time_interval(df, file_name, day, connection_ratio, file_dir=org_file_dir)
    unique_mmsi = df['MMSI'].unique()
    mmsi_to_new_id = {mmsi: idx for idx, mmsi in enumerate(unique_mmsi)}

    df['New_ID'] = df['MMSI'].map(mmsi_to_new_id)

    ship_data = {}
    for new_id in tqdm(df['New_ID'].unique(), desc="Filter Raw Data"):
        # print("new_id", new_id)
        ship_df = df[df['New_ID'] == new_id]
        filter_invalid_coordinates({
            'timestamps': ship_df[time_col_name].tolist(),
            'latitudes': ship_df[Lat_col_name].tolist(),
            'longitudes': ship_df[Lon_col_name].tolist()
        }, new_id, selected_interval, ship_data)

    recordNum = 0
    for ship_id, data in tqdm(ship_data.items(),  desc="Load Ship Data"):
        positions = zip(data['latitudes'], data['longitudes'])  
        ship_mbr = [min(data['latitudes']), min(data['longitudes']), \
            max(data['latitudes']), max(data['longitudes'])]
        positions_list = [pos for pos in positions]  
        mbr_list = [format_mbr((data['latitudes'][idx], data['longitudes'][idx], data['latitudes'][idx+1], data['longitudes'][idx+1])) for idx in range(len(data['latitudes'])-1)]
        positions = zip(data['latitudes'], data['longitudes'])  
        position_ids = [get_PositionID_based(pos) for pos in positions]  
        ship_data[ship_id]['positions'] = positions_list  
        ship_data[ship_id]['position_ids'] = position_ids  
        ship_data[ship_id]['ship_mbr'] = ship_mbr  
        ship_data[ship_id]['mbr_list'] = mbr_list  
        # print(ship_id, len(positions_list))
        recordNum += len(positions_list)
    print("The number of Record:", recordNum)

    return ship_data

def get_csv_files(directory):
    # 初始化一个空列表来存放CSV文件名
    csv_files = []
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        # 遍历当前目录下的所有文件
        for file in files:
            # 检查文件扩展名是否为.csv
            if file.lower().endswith('.csv'):
                # 将文件路径加入到列表中
                csv_files.append(os.path.join(root, file))

    return csv_files


def load_aisdk_dataset_df(datascalability=-1, org_file_dir = "./"):

    csv_file_list = get_csv_files(org_file_dir)
    csv_file = csv_file_list[0]

    df = pd.read_csv(csv_file)
    if (datascalability != -1):
        df = df.head(datascalability)

    return df

def load_trajectory_dataset_df(df, file_name, day, connection_ratio=0.99, org_file_dir = "./"):
    selected_interval = analysis_frequency_time_interval(df, file_name, day, connection_ratio, file_dir=org_file_dir)
    unique_mmsi = df['MMSI'].unique()
    mmsi_to_new_id = {mmsi: idx for idx, mmsi in enumerate(unique_mmsi)}

    df['New_ID'] = df['MMSI'].map(mmsi_to_new_id)

    ship_data = {}
    for new_id in tqdm(df['New_ID'].unique(), desc="Filter Raw Data"):
        # print("new_id", new_id)
        ship_df = df[df['New_ID'] == new_id]
        filter_invalid_coordinates({
            'timestamps': ship_df[time_col_name].tolist(),
            'latitudes': ship_df[Lat_col_name].tolist(),
            'longitudes': ship_df[Lon_col_name].tolist()
        }, new_id, selected_interval, ship_data)

    recordNum = 0
    for ship_id, data in tqdm(ship_data.items(),  desc="Load Ship Data"):
        positions = zip(data['latitudes'], data['longitudes'])
        ship_mbr = [min(data['latitudes']), min(data['longitudes']), \
            max(data['latitudes']), max(data['longitudes'])]
        positions_list = [pos for pos in positions]
        mbr_list = [format_mbr((data['latitudes'][idx], data['longitudes'][idx], data['latitudes'][idx+1], data['longitudes'][idx+1])) for idx in range(len(data['latitudes'])-1)]
        positions = zip(data['latitudes'], data['longitudes'])
        position_ids = [get_PositionID_based(pos) for pos in positions]
        ship_data[ship_id]['positions'] = positions_list
        ship_data[ship_id]['position_ids'] = position_ids
        ship_data[ship_id]['ship_mbr'] = ship_mbr
        ship_data[ship_id]['mbr_list'] = mbr_list
        # print(ship_id, len(positions_list))
        recordNum += len(positions_list)
    print("The number of Record:", recordNum)

    return ship_data

if __name__ == '__main__':
    load_aisdk_dataset(file_name='aisdk', day='20060302', org_file_dir='../Data', is_download=False)