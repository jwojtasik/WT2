import statsmodels.api as sm
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score

st.header('Kto ocalał z Titanica?')
st.image('titanic.png')
st.markdown('<p style="font-size: 10px;">Źródło grafiki: <a href = https://itg.com.pl/2023/12/23/history-and-data-science-the-titanic-kaggle-challenge/>https://itg.com.pl/2023/12/23/history-and-data-science-the-titanic-kaggle-challenge/</a></p>', unsafe_allow_html=True)
data = pd.read_csv(r'dane/titanic.csv', index_col = 0)
data = data.replace({'Sex': {'male': 1, 'female': 0}})

data_logit = data[['Survived', 'Age_scaled', 'Pclass', 'Sex']].dropna(axis = 0)

c0, c1, c2, c3, c4, c5 = st.columns([1,1.5,1.5,1.5,1.5,1])
with c1:
    w_age = st.number_input("Oszacuj wagę wieku\npasażera") # -2.95
with c3:
    w_class = st.number_input("Oszacuj wagę klasy podróży")# -1.28
with c2:
    w_gender = st.number_input("Oszacuj wagę płci\npasażera") # -2.52
with c4:
    w_const = st.number_input("Oszacuj wartość stałej") # 5.05

data_logit['prob'] = 1 /(1 + np.exp(-(w_age * data_logit['Age_scaled'] + w_class * data_logit['Pclass'] + w_gender * data_logit['Sex'] + w_const)))
data_logit['est_surv'] = (data_logit['prob'] > 0.5).astype(int)

st.markdown("Trafność stworzonego modelu wynosi")
st.markdown("<p style='text-align: center; font-size: 72px;'>{:.0%}</p>".format(accuracy_score(data_logit['est_surv'], data_logit['Survived'])), unsafe_allow_html=True)
st.markdown("Zmodyfikuj wagi aby poprawić trafność modelu!")


st.markdown('<p style="font-size: 10px;text-align: justify;">Projekt „Wiedza tAIemna 2.0” jest finansowany ze środków Europejskiej Fundacji Młodzieży. Niniejszy materiał został opracowany przez Instytut Przeciwdziałania Wykluczeniom przy wsparciu finansowym Europejskiej Fundacji Młodzieży Rady Europy. Wyrażone poglądy niekoniecznie odzwierciedlają oficjalne stanowisko Rady Europy.</p>', unsafe_allow_html=True)
c10, c11, c12, c13, c14 = st.columns([1,1.5,3,1.5,1])
with c11:
    st.image('EYF_visual_identity.png')
with c12:
    st.image('IPW_logo.png')
with c13:
    st.image('COE_logo.png')

