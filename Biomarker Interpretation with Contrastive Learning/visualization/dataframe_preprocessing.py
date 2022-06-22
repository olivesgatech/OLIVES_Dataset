
import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

def append_attributes(df_dir, df_start,save_dir):

    df_attr = pd.read_excel(df_dir)
    df_start = pd.read_csv(df_start)
    df_start['Eye_Data'] = ""
    df_start['Weight'] = ""
    df_start['Gender'] = ""
    #df_start['Race'] = ""
    df_start['Diabetes_Type'] = ""
    df_start['Diabetes_Years'] = ""
    df_start['HbAlC'] = ""
    df_start['Systemic Hypertension'] = ""
    df_start['Systolic BP'] = ""
    df_start['Diastolic BP'] = ""
    df_start['Smoking Status'] = ""
    df_start['Age'] = ""

    for i in tqdm(range(0,len(df_start))):
        patient = df_start.iloc[i,1]

        row = df_attr.loc[df_attr['Patient_ID'] == patient]


        if(row.empty):
            df_start.iloc[i,5:11] = -1
        else:
            df_start.iloc[i, 4] = row.iloc[0,1]
            df_start.iloc[i, 5] = row.iloc[0,2]
            df_start.iloc[i, 6] = row.iloc[0,5]
            df_start.iloc[i, 7] = row.iloc[0,6]
            df_start.iloc[i, 8] = row.iloc[0,7]
            df_start.iloc[i, 9] = row.iloc[0,9]
            df_start.iloc[i, 10] = row.iloc[0,10]
            df_start.iloc[i, 11] = row.iloc[0, 11]
            df_start.iloc[i, 12] = row.iloc[0, 12]
            df_start.iloc[i, 13] = row.iloc[0, 8]
            df_start.iloc[i, 14] = row.iloc[0, 4]

    df_data = df_start[df_start['Eye_side'] == df_start['Eye_Data']]
    df_data.to_csv(save_dir,index=False)

def train_test_split(df_start):
    df_start = pd.read_csv(df_start)
    df_filtered = df_start[df_start['Eye_side'] == df_start['Eye_Data']]
    test_df = df_filtered.sample(frac=.1,random_state = 32)
    train_df = df_filtered.drop(test_df.index)

    test_df.to_csv('./prime/test_attributes_cleaned.csv',index=False)
    train_df.to_csv('./prime/train_attributes_cleaned.csv', index=False)
import cv2 as cv2
def resize_prime_dataset(df_start):
    df_start = pd.read_csv(df_start)
    for i in tqdm(range(0, len(df_start))):

        dir =  df_start.iloc[i,0]

        im = Image.open(dir).convert("L")
        image = np.array(im)

        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        image = Image.fromarray(image)
        image.save(dir)

def construct_whole_dataframe(df_start,bcva,cst,leakage):
    df_start = pd.read_csv(df_start)
    bcva = pd.read_excel(bcva)
    cst = pd.read_excel(cst)

    df_start['BCVA'] = ""
    df_start['CST'] = ""

    for i in tqdm(range(0, len(df_start))):
        patient = df_start.iloc[i,1]
        week = df_start.iloc[i,2]
        row = bcva.loc[bcva['Patient_ID'] == patient]
        value = row.iloc[0][week]
        df_start.iloc[i,15] = value

    for i in tqdm(range(0, len(df_start))):
        patient = df_start.iloc[i,1]
        week = df_start.iloc[i,2]
        row = cst.loc[cst['Patient_ID'] == patient]
        value = row.iloc[0][week]
        df_start.iloc[i,16] = value

    df_start.to_csv('./recovery/overall_recovery.csv', index=False)


def analyze_dataframe(df_start):
    df = pd.read_csv(df_start)
    list  = df['DRSS'].tolist()
    u = (np.unique(list))
    print(len(u))
    print(u)
    print(df[df.DRSS == -1].shape[0])

def process_specific_attribute(df_start,attribute_type):
    at = attribute_type
    df = pd.read_csv(df_start)
    df_filtered = df[df['at'] == -1]
    df = df.drop(df_filtered.index)
    return df
def discretize_all_labels(df_start):
    df = pd.read_csv(df_start)

    for i in tqdm(range(0, len(df))):
        #Patient Processing
        patient = df.iloc[i, 1]
        num = int(patient[patient.find('-')+1:])
        df.iloc[i,1] = num
        # Age
        age = df.iloc[i,6]
        if(age >= 25 and age <=41):
            df.iloc[i, 6] = 0
        elif(age>41 and age <=56):
            df.iloc[i, 6] = 1
        else:
            df.iloc[i, 6] = 2
        # Gender
        gender = df.iloc[i,7]

        if(gender == 'M'):
            df.iloc[i,7] = 0
        else:
            df.iloc[i, 7] = 1
        # Race
        race = df.iloc[i,8]
        if(race == 'White'):
            df.iloc[i, 8] = 0
        elif(race == 'Black'):
            df.iloc[i, 8] = 1
        else:
            df.iloc[i,8] = 2
        # Diabetes_Type
        d_type = df.iloc[i,9]
        if(d_type == 1):
            df.iloc[i,9] = 0
        else:
            df.iloc[i,9] = 1
        # Diabetes_Years
        d_years = df.iloc[i,10]
        if(d_years >=0 and d_years<=11):
            df.iloc[i, 10] = 0
        elif(d_years>=12 and d_years<=22):
            df.iloc[i, 10] = 1
        else:
            df.iloc[i, 10] = 2
        # BMI
        bmi = df.iloc[i,11]
        if(bmi == 'nan'):
            df.iloc[i, 11] = -1
        elif(bmi>=15 and bmi<=28):
            df.iloc[i, 11] = 0
        elif (bmi >28 and bmi <= 38):
            df.iloc[i, 11] = 1
        else:
            df.iloc[i, 11] = 2
        #BCVA
        bcva = df.iloc[i, 12]
        if (bcva == 'nan'):
            df.iloc[i, 12] = -1
        elif (bcva >= 40 and bcva <= 60):
            df.iloc[i, 12] = 0
        elif (bcva > 60 and bcva <= 80):
            df.iloc[i, 12] = 1
        else:
            df.iloc[i, 12] = 2
        #DRSS
        drss=  df.iloc[i,13]
        if(drss=='nan'):
            df.iloc[i, 13] = -1
        elif(type(drss) == str):
            df.iloc[i, 13] = -1
        elif(drss >=43 and drss <=47):
            df.iloc[i, 13] = 0
        elif (drss >= 43 and drss <= 61):
            df.iloc[i, 13] = 1
        else:
            df.iloc[i, 13] = 2
        # CST
        cst = df.iloc[i, 14]
        if (cst == 'nan'):
            df.iloc[i, 14] = -1
        elif (cst >= 192 and cst <= 260):
            df.iloc[i, 14] = 0
        elif (cst > 260 and cst <=316):
            df.iloc[i, 14] = 1
        else:
            df.iloc[i, 14] = 2
        #Leakage Index
        li = df.iloc[i, 15]
        if (li == 'nan'):
            df.iloc[i, 15] = -1
        elif (li >= 0 and li <= 3.44):
            df.iloc[i, 15] = 0
        elif (li > 3.44 and li <= 6.88):
            df.iloc[i, 15] = 1
        else:
            df.iloc[i, 15] = 2

    df.to_csv('./prime/complete_data_discretized.csv',index=False)
def dataframe_cleaning_recovery(df_start):
    df = pd.read_csv(df_start)
    for i in tqdm(range(0, len(df))):
        patient = df.iloc[i, 1]
        num = int(patient[patient.find('-') + 1:])
        df.iloc[i, 1] = num

        bcva = df.iloc[i, 15]
        if (math.isnan(bcva)):
            df.iloc[i, 15] = -1
        else:
            df.iloc[i, 15] = int(bcva)
        # CST
        cst = df.iloc[i, 16]
        if (math.isnan(cst)):
            df.iloc[i, 16] = -1
        else:
            df.iloc[i, 16] = int(cst)
    df_clean_bcva = df[df['BCVA'] !=-1]
    df_clean_bcva.to_csv('./recovery/recovery_bcva_clean.csv',index=False)
    df_clean_cst = df[df['CST'] != -1]
    df_clean_cst.to_csv('./recovery/recovery_cst_clean.csv', index=False)
def dataframe_processing_recovery(df_start):
    df = pd.read_csv(df_start)
    for i in tqdm(range(0, len(df))):
        gender = df.iloc[i,6]
        dt = df.iloc[i,7]
        hyper = df.iloc[i,10]
        ss = df.iloc[i,13]
        if(gender == 'Male'):
            df.iloc[i,6] = 1
        else:
            df.iloc[i,6] = 0

        if (dt == 'I'):
            df.iloc[i, 7] = 0
        else:
            df.iloc[i, 7] = 1

        if (hyper == 'Yes'):
            df.iloc[i, 10] = 0
        else:
            df.iloc[i, 10] = 1

        if (ss == 'Never'):
            df.iloc[i, 13] = 0
        elif(ss == 'Current'):
            df.iloc[i, 13] = 1
        else:
            df.iloc[i, 13] = 2

    df.to_csv('./recovery/recovery_attributes_converted.csv',index=False)

def dataframe_cleaning(df_start):
    df = pd.read_csv(df_start)
    x= float('nan')

    for i in tqdm(range(0, len(df))):
        # Patient Processing
        patient = df.iloc[i, 1]
        num = int(patient[patient.find('-') + 1:])
        df.iloc[i, 1] = num
        # Gender
        gender = df.iloc[i, 7]

        if (gender == 'M'):
            df.iloc[i, 7] = 0
        else:
            df.iloc[i, 7] = 1
        # Race
        race = df.iloc[i, 8]
        if (race == 'White'):
            df.iloc[i, 8] = 0
        elif (race == 'Black'):
            df.iloc[i, 8] = 1
        else:
            df.iloc[i, 8] = 2
        # Diabetes_Type
        d_type = df.iloc[i, 9]
        if (d_type == 1):
            df.iloc[i, 9] = 0
        else:
            df.iloc[i, 9] = 1
        #BMI
        bmi = df.iloc[i, 11]
        if (math.isnan(bmi)):
            df.iloc[i, 11] = -1
        else:
            df.iloc[i, 11] = int(bmi)
        #BCVA
        bcva = df.iloc[i, 12]
        if (math.isnan(bcva)):
            df.iloc[i, 12] = -1
        else:
            df.iloc[i,12] = int(bcva)
        # CST
        cst = df.iloc[i, 14]
        if (math.isnan(cst)):
            df.iloc[i, 14] = -1
        else:
            df.iloc[i,14] = int(cst)

        # Leakage Index
        li = df.iloc[i, 15]
        if (math.isnan(li)):
            df.iloc[i, 15] = -1
        else:
            df.iloc[i,15] = int(li)

        drss = df.iloc[i, 13]

        if (type(drss) == str):
            if(len(drss)>2 and drss[0]!='V'):
                drss = float(drss)
            elif (drss[0] == 'V'):
                drss = -1
            else:
                drss = int(drss)
        elif(math.isnan(drss)):
            drss = -1
        else:
            drss = int(drss)
        #print(drss)
        df.iloc[i, 13] = int(drss)
    df.to_csv('./prime/complete_data_cleaned.csv', index=False)



def prime_biomarker_processing(dir_start):
    files = os.listdir(dir_start)
    os.chdir(dir_start)
    full_df = []
    for file in files:

        df_s1 = pd.read_excel(file,sheet_name=0)
        df_s2 = pd.read_excel(file, sheet_name=1)

        df = pd.concat([df_s1,df_s2])

        for i in range(0,len(df)):
            header = '/data/Datasets/Prime_FULL'
            file_path = df.iloc[i,0]
            scan_num = df.iloc[i,1]

            split_path = file_path.split('/')
            img_name = split_path[4]
            img_name_split = img_name.split('.')
            img_name_split[0] = str(scan_num)

            new_img_name = img_name_split[0] + '.' + img_name_split[1]

            split_path[0] = header
            split_path[4] = new_img_name
            if(split_path[1] == '01-017'):
                split_path[1] = '02-032'
            new_file_path = os.path.join(split_path[0],split_path[1],split_path[2],split_path[3])
            df.iloc[i,0] = new_file_path
        full_df.append(df)

    f = pd.concat(full_df)
    for i in range(0,len(f)):
        file_path = f.iloc[i,0]
        scan_num = f.iloc[i,1]-1
        os.chdir(file_path)
        files = os.listdir(file_path)
        format = files[0][-3:]
        img_name = str(scan_num) + '.' + format
        #print(img_name)
        f.iloc[i,0] = os.path.join(file_path,img_name)

    print(f.head())

    os.chdir('/home/kiran/Desktop/Dev/SupCon')
    f.to_csv('./prime/biomarkers.csv',index=False)


def analyze_biomarkers(df_start):

    df= pd.read_csv(df_start)
    counts = df['Fluid (IRF)'].value_counts()

    print(counts)

def split_bio_supcon(df_bio,df_supcon):
    df_bio = pd.read_csv(df_bio)
    df_supcon = pd.read_csv(df_supcon)
    appended_data = []

    for i in tqdm(range(0,len(df_bio))):

        file_path_2 = df_bio.iloc[i,0]


        df_supcon = df_supcon[df_supcon['File_Path']!=file_path_2]

    df_filtered = df_supcon[df_supcon['Eye_side'] == df_supcon['Eye_Data']]
    print(len(df_filtered))
    df_filtered.to_csv('./prime/train_prime_full_train.csv',index=False)

def biomarker_split(bio_df,att):
    bio_df = pd.read_csv(bio_df)
    count = bio_df[att].value_counts()
    bio_df_0 = bio_df[bio_df[att] == 0]
    bio_df_1 = bio_df[bio_df[att] == 1]

    test_0 = bio_df_0.sample(n=500,random_state=32)
    test_1 = bio_df_1.sample(n=500, random_state=32)
    test = pd.concat([test_0,test_1])
    train = bio_df.drop(test.index)
    val_0 = test[test[att] == 0]
    val_1 = test[test[att] == 1]
    val_0 = val_0.sample(n=250,random_state=32)
    val_1 = val_1.sample(n=250,random_state=32)
    val = pd.concat([val_0,val_1])
    test= test.drop(val.index)
    test.to_csv('./prime/biomarker_patientreduction_training/test_fluirf.csv',index=False)
    train.to_csv('./prime/biomarker_patientreduction_training/train_fluirf.csv', index=False)
    val.to_csv('./prime/biomarker_patientreduction_training/val_fluirf.csv',index=False)

def biomarker_append_attributes(bio_df,att_df,name):
    bio_df = pd.read_csv(bio_df)
    bio_df['Diabetes_Type'] = ''
    bio_df['BCVA'] = ''
    bio_df['DRSS'] = ''
    bio_df['CST'] = ''
    bio_df['Leakage_Index'] = ''

    att_df = pd.read_csv(att_df)
    for i in tqdm(range(0,len(bio_df))):

        file_path_2 = bio_df.iloc[i,0]


        row = att_df[(att_df==file_path_2).any(axis=1)]
        #print(row)
        bio_df.iloc[i,22] = row.iloc[0,9]
        bio_df.iloc[i,23] = row.iloc[0,12]
        bio_df.iloc[i,24] = row.iloc[0,13]
        bio_df.iloc[i,25] = row.iloc[0, 14]
        bio_df.iloc[i,26] = row.iloc[0, 15]
    bio_df.to_csv(name,index=False)
def biomarker_patient_removal(df_bio,df_prime):
    df_bio = pd.read_csv(df_bio)
    df_prime = pd.read_csv(df_prime)
    print(df_bio['Patient_ID'].unique())
    df_prime_patient_reduced = df_prime[(df_prime['Patient_ID'] == 8) | (df_prime['Patient_ID'] == 31) |
                                        (df_prime['Patient_ID'] == 1) | (df_prime['Patient_ID'] == 2) |
                                        (df_prime['Patient_ID'] == 12) | (df_prime['Patient_ID'] == 13) |
                                        (df_prime['Patient_ID'] == 14) | (df_prime['Patient_ID'] == 20) |
                                        (df_prime['Patient_ID'] == 23) | (df_prime['Patient_ID'] == 25)]
    print(len(df_prime_patient_reduced))
    for i in tqdm(range(0, len(df_bio))):
        file_path_2 = df_prime_patient_reduced.iloc[i, 0]
        df_bio = df_bio[df_bio['Path (Trial/Folder/Week/Eye/Image Name)'] != file_path_2]


def extract_patient_data(df_start):
    df = pd.read_csv(df_start)
    df['Patient_ID'] = ""
    for i in tqdm(range(0,len(df))):
        file_path = df.iloc[i,0]
        split_path = file_path.split('/')
        patient = split_path[4]
        num = int(patient[patient.find('-')+1:])
        df.iloc[i,27] = num
    df.to_csv(df_start,index=False)
def split_percentage(df_start,percentage_amount,new_path):
    df = pd.read_csv(df_start)
    df_sampled = df.sample(frac = percentage_amount,random_state=1)
    file_name = new_path + str(int(100*percentage_amount)) + '.csv'
    df_sampled.to_csv(file_name,index=False)

def create_prime_drss(df_start):
    df = pd.read_csv(df_start)
    df_new = df
    df = df[df.DRSS !=-1]
    df.to_csv('./prime/train_prime_full_drss_correct.csv', index = False)

def discretize_labels_prime(df_start,num_cuts):
    df = pd.read_csv(df_start)
    df['BCVA'] = pd.qcut(df['BCVA'],q=num_cuts,labels=False)
    df['CST'] = pd.qcut(df['CST'], q=num_cuts, labels=False)
    df['Diabetes_Years'] = pd.qcut(df['Diabetes_Years'], q=num_cuts, labels=False)
    df['Age'] = pd.qcut(df['Age'], q=num_cuts, labels=False)
    file_name = './discretized_labels/' + 'prime_' +str(num_cuts) +'.csv'
    df.to_csv(file_name,index=False)
    print(df.head())
def discretize_labels_recovery(df_start,num_cuts):
    df = pd.read_csv(df_start)
    df['BCVA'] = pd.qcut(df['BCVA'],q=num_cuts,labels=False)
    df['CST'] = pd.qcut(df['CST'], q=num_cuts, labels=False)
    df['Diabetes_Years'] = pd.qcut(df['Diabetes_Years'], q=num_cuts, labels=False,duplicates='drop')
    df['Age'] = pd.qcut(df['Age'], q=num_cuts, labels=False)
    df['Systolic BP'] = pd.qcut(df['Systolic BP'], q=num_cuts, labels=False)
    df['Diastolic BP'] = pd.qcut(df['Diastolic BP'], q=num_cuts, labels=False)
    df['HbAlC'] = pd.qcut(df['HbAlC'], q=num_cuts, labels=False)
    file_name = './discretized_labels/' + 'recovery_' +str(num_cuts) +'.csv'
    df.to_csv(file_name,index=False)
    print(df.head())
def patient_split(df_train):
    df = pd.read_csv(df_train)
    df['Patient_ID'] = ""
    for i in tqdm(range(0, len(df))):
        file_path = df.iloc[i, 0]
        split_path = file_path.split('/')
        patient = split_path[4]
        num = int(patient[patient.find('-') + 1:])
        # print(num)
        df.iloc[i, 27] = num
    df_subset = df.loc[(df['Patient_ID'] == 8) | (df['Patient_ID'] == 31)
                       | (df['Patient_ID'] == 1) | (df['Patient_ID'] == 2)
                       | (df['Patient_ID'] == 12) | (df['Patient_ID'] == 13)
                        | (df['Patient_ID'] == 14) | (df['Patient_ID'] == 20) | (df['Patient_ID'] == 23) | (df['Patient_ID'] == 25)]

    df_subset_1 = df.loc[(df['Patient_ID'] == 26) | (df['Patient_ID'] == 27)
                       | (df['Patient_ID'] == 28) | (df['Patient_ID'] == 35)
                       | (df['Patient_ID'] == 38) | (df['Patient_ID'] == 37)
                       | (df['Patient_ID'] == 40) | (df['Patient_ID'] == 47) | (df['Patient_ID'] == 48) | (
                                   df['Patient_ID'] == 4)]
    df_subset_2 = df.loc[(df['Patient_ID'] == 5) | (df['Patient_ID'] == 10)
                         | (df['Patient_ID'] == 15) | (df['Patient_ID'] == 16)
                         | (df['Patient_ID'] == 17) | (df['Patient_ID'] == 18)
                         | (df['Patient_ID'] == 19) | (df['Patient_ID'] == 24) | (df['Patient_ID'] == 29) | (
                                 df['Patient_ID'] == 30)]
    df_subset_3 = df.loc[(df['Patient_ID'] == 32) | (df['Patient_ID'] == 34)
                         | (df['Patient_ID'] == 36) | (df['Patient_ID'] == 39)
                         | (df['Patient_ID'] == 41) | (df['Patient_ID'] == 42)
                         | (df['Patient_ID'] == 43) | (df['Patient_ID'] == 45) | (df['Patient_ID'] == 44) | (
                                 df['Patient_ID'] == 46)]

    counts = df_subset_1['Fully attached vitreous face'].value_counts()
    train_df = df.drop(df_subset_1.index)
    counts = train_df['Fully attached vitreous face'].value_counts()
    bio_0 = df_subset_1[df_subset_1['Fully attached vitreous face'] == 0]
    bio_1 = df_subset_1[df_subset_1['Fully attached vitreous face'] == 1]
    val_0 = bio_0.sample(n=410, random_state=32)
    val_1 = bio_1.sample(n=410, random_state=32)
    test_df = pd.concat([val_0,val_1])
    counts = test_df['Fully attached vitreous face'].value_counts()
    train_df.to_csv('./prime/patient_separated_biomarkers/patient_splits/train_biomarkers_1.csv',index=False)
    test_df.to_csv('./prime/patient_separated_biomarkers/patient_splits/test_fitvat_1.csv',index=False)

def check(df_start):
    df = pd.read_csv(df_start)
    counts = df['Fluid (IRF)'].value_counts()
    print(counts)


def gen_128(df_start):
    df = pd.read_csv(df_start)
    for i in tqdm(range(0, len(df_start))):

        dir =  df.iloc[i,0]
        list = dir.split('/')

        list[3] = 'Prime_FULL_128'

        dir_new = os.path.join(list[1],list[2],list[3],list[4],list[5],list[6],list[7])

        dir_new = '/' + dir_new
        df.iloc[i,0]=dir_new
    df.to_csv('./prime/train_prime_full_128.csv',index=False)

########################### Functions to format csv files for continuity ##############################################


def remove_recovery(df_dir):
    df = pd.read_csv(df_dir)


def remove_data_datasets(df_dir):
    df = pd.read_csv(df_dir)
    for i in range(0,len(df)):
        path = df.iloc[i,0]
        path_new = path[14:]
        df.iloc[i,0] = path_new

    df.to_csv(df_dir,index=False)

def add_patient_id(df_dir):
    df = pd.read_csv(df_dir)
    df['Patient_ID'] = ""
    for i in tqdm(range(0, len(df))):
        patient = df.iloc[i,1]
        code = int(patient[0:4])
        df.iloc[i,8] = code



    df.to_csv(df_dir,index=False)

def fix_bmi_leakage_index(bio_df_dir,att_df_dir):
    bio_df = pd.read_csv(bio_df_dir)

    att_df = pd.read_csv(att_df_dir)
    for i in tqdm(range(0,len(att_df))):

        file_path_2 = att_df.iloc[i,0]


        row = bio_df[(bio_df==file_path_2).any(axis=1)]
        if(row.empty == False):
            att_df.iloc[i,15] = row.iloc[0,15]

            att_df.iloc[i, 11] = row.iloc[0,11]

    att_df.to_csv(att_df_dir,index=False)



if __name__ == '__main__':
    df_dir = '/home/kiran/Desktop/Dev/SupCon_OCT_Clinical/clinical_datasets/recovery/overall_recovery_complete.csv'
    remove_data_datasets(df_dir)
