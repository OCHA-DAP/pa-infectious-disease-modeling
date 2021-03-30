import matplotlib.pyplot as plt
import pandas as pd
import os

from pandas.core import base
dir_path = os.path.dirname(os.path.realpath(__file__))

for disease in ["Cholera","Malaria","Measles"]:
    baseline=pd.read_excel(f'{dir_path}/Results_Baseline_{disease}_National.xlsx')
    covid=pd.read_excel(f'{dir_path}/Results_COVID_{disease}_National.xlsx')
    all=pd.merge(left=baseline,right=covid,how='left',left_on='time',right_on='time',\
        suffixes=[' - Baseline',' - COVID scenario'])
    all=all.filter(regex='Infected')
    # print(all.columns)
    all.plot(title=disease)
    plt.legend()
    plt.savefig(f"{dir_path}/Comparison_{disease}.png")

plt.show()