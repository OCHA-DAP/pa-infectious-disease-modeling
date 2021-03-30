import matplotlib.pyplot as plt
import pandas as pd
import os

from pandas.core import base
dir_path = os.path.dirname(os.path.realpath(__file__))
COVID_impact={
    "Cholera":{
        "alpha":1.2,
        "beta1":1.2,
        "beta2":1.6,
        "delta":0.8,
        "bhat":0.8
    },
    "Malaria":{
        "gamma":1.2,
        "zeta":1.4,
        "a":1.2
    },
    "Measles":{
        "kappa1":0.8,
        "kappa2":0.8
    }
}


for disease in ["Cholera","Malaria","Measles"]:
    baseline_scenario=pd.read_excel(f'{dir_path}/Inputs_Baseline_{disease}_National.xlsx')
    COVID_scenario=baseline_scenario.copy(deep=True)
    print(f'\n Generating COVID scenario for {disease}')
    # print(COVID_scenario)
    for key,value in COVID_impact[disease].items():
        if len(COVID_scenario.loc[COVID_scenario['Parameter']==key])==0:
            print(f'WARNING missing parameter {key}')
        COVID_scenario.loc[COVID_scenario['Parameter']==key,[200]]=\
            COVID_scenario.loc[COVID_scenario['Parameter']==key,[200]] * value
        COVID_scenario.to_excel(f'{dir_path}/Inputs_COVID_{disease}_National.xlsx',index=False)


