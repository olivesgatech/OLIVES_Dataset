import os
import pandas as pd

from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def generate_fundus(csv_file,target_dir):
    df = pd.read_csv(csv_file)

    for i in tqdm(range(0,len(df))):
        path = df.iloc[i,0]
        split = path.split('/')
        if(split[0] == 'Prime_FULL'):
            fundus_file_name = 'fundus_' + split[3] + "_" + split[2] + '.tif'

            split[-1] = fundus_file_name
            path = os.path.join(*split)
            df.iloc[i,0] = os.path.join(*split)
        else:
            fundus_file_name = 'fundus_' + split[4] + "_" + split[3] + '.tif'
            split[-1] = fundus_file_name
            df.iloc[i, 0] = os.path.join(*split)
    df = df.drop_duplicates()
    df.to_csv(target_dir,index=False)

def prime_compressed_test(csv_file,target_dir):
    df = pd.read_csv(csv_file)
    df = df[['File_Path', 'BCVA', 'CST','Eye_ID','Patient_ID']]
    df.to_csv(target_dir,index=False)


def trex_compressed_test(csv_file,target_dir):
    df = pd.read_csv(csv_file)
    df = df[['File_Path', 'BCVA', 'CST', 'Eye_ID', 'Patient_ID']]
    df.to_csv(target_dir,index=False)

def combine_prime_trex_test(prime_file,trex_file,target_dir):
    df_prime = pd.read_csv(prime_file)
    df_trex = pd.read_csv(trex_file)
    df_target = pd.concat([df_prime,df_trex])

    df_target.to_csv(target_dir, index=False)

def attach_treatment_label(csv_file,target_dir):
    df = pd.read_csv(csv_file)
    df['Week'] = ""
    df_new = pd.DataFrame(columns = ['File_Path','BCVA','CST','Eye_ID','Patient_ID','Week','Treatment'])
    dataframe_list = []
    # Attach Week Number
    for i in range(0,len(df)):
        path = df.iloc[i,0]
        split = path.split('/')
        week = split[-3]
        week_num = int(week[1:])
        df.iloc[i,5] = week_num

    eye_list = df['Eye_ID'].unique()
    # Extract individual eye dataframe
    for j in range(0,len(eye_list)):
        df_eye = df.loc[df['Eye_ID'] == eye_list[j]]
        df_eye = df_eye.sort_values(by='Week')
        df_eye['Treatment'] = ""
        for k in range(0,len(df_eye)):
            if(k < len(df_eye) - 1):
                cur = df_eye.iloc[k,1]
                next = df_eye.iloc[k+1,1]
                if(next>cur):
                    df_eye.iloc[k,6] = 1
                else:
                    df_eye.iloc[k, 6] = 0
            else:
                df_eye.iloc[k, 6] = -1
        dataframe_list.append(df_eye)
    df_final = pd.concat(dataframe_list)
    df_final = df_final[df_final.Treatment != -1]
    df_final.to_csv(target_dir,index=False)

def show_value_count(csv_file):
    df = pd.read_csv(csv_file)

    print(df['Treatment'].value_counts())


    return 1

def generate_3D_Volume(csv_file,target_dir):
    df = pd.read_csv(csv_file)
    df['Week'] = ""
    df['Scan'] = ""
    dataframe_list = []
    volume_df = pd.DataFrame(columns = ['File_Path','BCVA','CST','Eye_ID','Patient_ID'])
    # Attach Week Number
    for i in tqdm(range(0,len(df))):
        path = df.iloc[i,0]
        split = path.split('/')
        week = split[-3]
        week_num = int(week[1:])
        df.iloc[i,5] = week_num
    #Attach Scan Number
    for i in tqdm(range(0,len(df))):
        path = df.iloc[i,0]
        split = path.split('/')
        image_name = split[-1]
        if(split[1] == 'Prime_FULL'):
            file_split = image_name.split('.')
            scan_num = int(file_split[0])
            df.iloc[i,6] = scan_num
        elif(split[1] == 'TREX DME'):
            file_split = image_name.split('.')

            scan_num = int(file_split[0][-4:])
            df.iloc[i,6] = scan_num
    eye_list = df['Eye_ID'].unique()
    week_list = df['Week'].unique()
    print(df.head())


    # Extract individual eye dataframe

    for j in tqdm(range(0,len(eye_list))):
        for k in range(0,len(week_list)):
            df_eye_week = df.loc[df['Eye_ID'] == eye_list[j]]
            df_eye_week = df_eye_week.loc[df_eye_week['Week'] == week_list[k]]
            df_eye_week = df_eye_week.sort_values(by='Scan')
            volume_list = []
            for w in range(0,len(df_eye_week)):
                if(w<=45):
                    image = Image.open('/data/Datasets'+df_eye_week.iloc[w,0]).convert("L")
                    image = image.resize((224,224))
                    image = np.asarray(image)
                    volume_list.append(image)
            if(df_eye_week.empty == False):
                volume_array = np.array(volume_list)
                split = df_eye_week.iloc[0,0].split('/')
                split[-1] = 'volume_' + str(df_eye_week.iloc[0,3]) + '_' + str(df_eye_week.iloc[0,5]) + '.npy'
                numpy_name = '/data/Datasets/'+ os.path.join(*split)
                np.save(numpy_name,volume_array)
               # volume_df.loc[len(volume_df)] = [numpy_name,df_eye_week.iloc[0,1],df_eye_week.iloc[0,2],df_eye_week.iloc[0,3],df_eye_week.iloc[0,4]]

    #volume_df.to_csv(target_dir,index=False)

def time_series_generation(csv_file,target_dir):
    df = pd.read_csv(csv_file)
    df['Week'] = ""
    # Attach Week Number
    for i in range(0, len(df)):
        path = df.iloc[i, 0]
        split = path.split('/')
        week = split[-3]
        week_num = int(week[1:])
        df.iloc[i, 5] = week_num
    sequence_df = pd.DataFrame(columns=['File_Path', 'BCVA', 'CST', 'Eye_ID', 'Patient_ID','Treatment_Final'])
    eye_list = df['Eye_ID'].unique()
    # Extract individual eye dataframe
    for j in tqdm(range(0,len(eye_list))):
        df_eye = df.loc[df['Eye_ID'] == eye_list[j]]
        df_eye = df_eye.sort_values(by='Week')
        start = df_eye.iloc[0,1]
        end = df_eye.iloc[len(df_eye)-1,1]


        sequence_list = []
        for w in range(0, len(df_eye)-1):
            image = np.load('/data/Datasets' + df_eye.iloc[w, 0])
            #image = image.resize((224, 224))
            #image = np.asarray(image)
            sequence_list.append(image)

        sequence_array = np.array(sequence_list)
        print(sequence_array.shape)
        split = df_eye.iloc[0, 0].split('/')
        if(split[0] == 'Prime_FULL'):
            path = '/data/Datasets/' + os.path.join(*split[0:2])
            file = path + '/sequence_3d_' + str(df_eye.iloc[0,3]) + '.npy'
            print(file)
            if(start<end):
                sequence_df.loc[len(sequence_df)] = [file, df_eye.iloc[0, 1], df_eye.iloc[0, 2],
                                             df_eye.iloc[0, 3], df_eye.iloc[0, 4],1]
            else:
                sequence_df.loc[len(sequence_df)] = [file, df_eye.iloc[0, 1], df_eye.iloc[0, 2],
                                                     df_eye.iloc[0, 3], df_eye.iloc[0, 4], 0]
            np.save(file, sequence_array)
        else:
            path = '/data/Datasets/' + os.path.join(*split[0:3])
            file = path + '/sequence_3d_' + str(df_eye.iloc[0, 3]) + '.npy'
            print(file)
            if (start < end):
                sequence_df.loc[len(sequence_df)] = [file, df_eye.iloc[0, 1], df_eye.iloc[0, 2],
                                                     df_eye.iloc[0, 3], df_eye.iloc[0, 4], 1]
            else:
                sequence_df.loc[len(sequence_df)] = [file, df_eye.iloc[0, 1], df_eye.iloc[0, 2],
                                                     df_eye.iloc[0, 3], df_eye.iloc[0, 4], 0]

            np.save(file, sequence_array)
    sequence_df.to_csv(target_dir,index=False)

def remove_data_datasets(df_dir):
    df = pd.read_csv(df_dir)
    for i in range(0,len(df)):
        path = df.iloc[i,0]
        path_new = path[14:]
        df.iloc[i,0] = path_new

    df.to_csv(df_dir,index=False)

def fix_cols(csv_file,target_dir):
    df = pd.read_csv(csv_file)
    df = df.drop('Week',axis=1)
    df = df.drop('Treatment', axis=1)
    df.to_csv(target_dir, index=False)

def generate_last_csv(csv_file,target_dir):
    df = pd.read_csv(csv_file)
    df['Week'] = ""
    df_new = pd.DataFrame(columns = ['File_Path','BCVA','CST','Eye_ID','Patient_ID','Week','Treatment'])
    # Attach Week Number
    for i in range(0, len(df)):
        path = df.iloc[i, 0]
        split = path.split('/')
        week = split[-3]
        week_num = int(week[1:])
        df.iloc[i, 5] = week_num

    eye_list = df['Eye_ID'].unique()
    # Extract individual eye dataframe
    for j in range(0, len(eye_list)):
        df_eye = df.loc[df['Eye_ID'] == eye_list[j]]
        df_eye = df_eye.sort_values(by='Week')
        df_eye['Treatment'] = ""
        bcva_1 = df_eye.iloc[0,1]
        bcva_n = df_eye.iloc[len(df_eye)-1,1]
        treat = 0
        if(bcva_1 < bcva_n):
            df_new.loc[len(df_new)] = [df_eye.iloc[0,0],df_eye.iloc[0,1],df_eye.iloc[0,2],df_eye.iloc[0,3],df_eye.iloc[0,4],df_eye.iloc[0,5],1]
        else:
            df_new.loc[len(df_new)] = [df_eye.iloc[0, 0], df_eye.iloc[0, 1], df_eye.iloc[0, 2], df_eye.iloc[0, 3],
                                       df_eye.iloc[0, 4], df_eye.iloc[0, 5], 0]
    df_new.to_csv(target_dir,index=False)

def average_weeks(csv_file):
    df = pd.read_csv(csv_file)
    df['Week'] = ""
    # Attach Week Number
    for i in range(0, len(df)):
        path = df.iloc[i, 0]
        split = path.split('/')
        week = split[-3]
        week_num = int(week[1:])
        df.iloc[i, 5] = week_num

    eye_list = df['Eye_ID'].unique()
    sum = 0
    total = 0
    for j in range(0, len(eye_list)):
        df_eye = df.loc[df['Eye_ID'] == eye_list[j]]
        df_eye = df_eye.sort_values(by='Week')
        sum = sum + len(df_eye)
        total = total + 1
    print(sum/total)
if __name__ == '__main__':
    csv_file = '/home/kiran/Desktop/Dev/OLIVES_Dataset/treatment_prediction_datasets/fundus_dir/fundus.csv'
    average_weeks(csv_file)