


import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import matplotlib
def treatment_graphs(csv_file):
    df = pd.read_csv(csv_file)
    df_pat = df[df['Eye_ID'] == 23]
    index_list = []
    sum_list = []
    total = 0
    index_list.append(0)
    sum_list.append(0)
    for i in range(0,len(df_pat)):
        index_list.append(i+1)
        if(df_pat.iloc[i,6] == 1):
            total = total + 1
        else:
            total = total - 1
        sum_list.append(total)
    X_axis = np.arange(len(index_list))
    np.arange(len(index_list))
    plt.bar(X_axis-.2,sum_list,.4,label = 'Patient 23')

    df_pat = df[df['Eye_ID'] == 72]
    index_list = []
    sum_list = []
    total = 0
    index_list.append(0)
    sum_list.append(0)
    for i in range(0,len(df_pat)):
        index_list.append(i+1)
        if(df_pat.iloc[i,6] == 1):
            total = total + 1
        else:
            total = total - 1
        sum_list.append(total)
    X_axis = np.arange(len(index_list))
    plt.bar(X_axis+ 0.2,sum_list,.4, label = 'Patient 72')
    plt.grid()
    plt.title('Patient Treatment Progression')
    plt.xlabel('Visit Number')
    plt.ylabel('Treatment Results')
    plt.legend()
    plt.show()

def treatment_graphs_calibrated(csv_file):
    df = pd.read_csv(csv_file)
    df['Week'] = ""

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

        index_list = []
        index_list.append(0)
        sum_list = []
        sum_list.append(0)
        total = 0
        for k in range(0,len(df_eye)):
            if(k!= len(df_eye)-1):
                index_list.append(k)
                bcva_1 = df_eye.iloc[k,1]
                bcva_2 = df_eye.iloc[k+1,1]

                diff = bcva_2 - bcva_1
                total = total + diff
                sum_list.append(total)
        X_axis = np.arange(len(index_list))
        if(j==0):
            plt.bar(X_axis + .2, sum_list, .4,label = 'Patient ' + str(df_eye.iloc[0,4]))
        else:
            plt.bar(X_axis - .2, sum_list, .4,label = 'Patient ' + str(df_eye.iloc[0,4]))

        if(j==1):
            break


    plt.grid()
    plt.title('Treatment Progression Ground Truth')
    plt.xlabel('Visit Number')
    plt.ylabel('BCVA increase compared to first week')
    plt.legend()
    plt.show()

def generate_polar_plot(csv_file):
    df = pd.read_csv(csv_file)
    eye_list = df['Eye_ID'].unique()
    degrees_total = 0
    for j in range(0, len(eye_list)):
        df_eye = df.loc[df['Eye_ID'] == eye_list[j]]
        df_eye = df_eye.sort_values(by='Week')
        degrees = 180/len(df_eye)
        if(j==25):
            for k in range(0,len(df_eye)):
                treat = df.iloc[k,6]
                print(treat)
                if(treat == 0):
                    degrees_total += -1*degrees
                else:
                    degrees_total +=degrees
            break
    print(degrees_total)
    if(degrees_total < 0):
        degrees_total = 360 + degrees_total
    plt.rc('grid', color='black', linewidth=1, linestyle='-')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    # force square figure and square axes looks better for polar, IMO
    width, height = plt.rcParams['figure.figsize']
    size = min(width, height)
    print(size)
    # make a square figure
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ############ Colorbar


    arr2 = plt.arrow(degrees_total / 180. * np.pi, .2, 0, .4, alpha=0.5, width=0.08,
                     edgecolor='black', facecolor='black', lw=1, zorder=5)
    ax.patch.set_facecolor((degrees_total/360, .25,.5))
    ax.patch.set_alpha(.8)

    ax.set_rmax(1.0)
    plt.title('Polar Representation of Disease Progression')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    csv_file = '/home/kiran/Desktop/Dev/NeurIPS_2022_Dataset/treatment_prediction_datasets/fundus_dir/fundus_treatment.csv'
    #treatment_graphs_calibrated(csv_file)
    generate_polar_plot(csv_file)