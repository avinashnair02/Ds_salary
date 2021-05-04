import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import sklearn
# imported the required dependicies

df=pd.read_csv('salary_data_cleaned.csv')

st.title('Data Scientist Salary Predictor')
st.title('Covid-19')

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1513530534585-c7b1394c6d51?ixlib=rb-1.2.1&ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&auto=format&fit=crop&w=1351&q=80");
background-repeat: no-repeat;
background-size: cover;

}
.block-container {
    backdrop-filter: blur(10px);
}
.markdown-text-container > h1{text-align:center}
</style>

'''

st.markdown(page_bg_img, unsafe_allow_html=True)


#model=joblib.Model('model.pkl')
#loaded_model = pickle.load(open(finalized_model.sav, 'rb'))

def load_models():
    file_name = "model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

st.markdown(
    "**All the fields  are mandatory")

st.subheader('Company Details: \n Check Glassdoor for exact values, if unsure')
rating = st.slider('Glassdoor Rating of the Company',
                   min_value=0.0, max_value=5.0, step=0.1)
age = st.number_input('Age of the Company', step=1.0, min_value=0.0)
num_comp = st.number_input('Number of Competitors', step=1.0, min_value=0.0)

st.subheader('Details about the Job:')

jobhq = st.radio(
    "Is the Job at Headquarters? (0 for No, 1 for Yes)", options=[0, 1])
job_type_num = st.selectbox("Job State",
                            options=df["job_state"].unique())
        
def number_simplifier(role):
    if role == "data scientist":
        return 3
    elif role == "data engineer":
        return 2
    elif role == "analyst":
        return 1
    elif role == "director":
        return 4
    elif role == "manager":
        return 5
    elif role == "mle":
        return 6
    elif role == "na":
        return 7


job_type_num1 = number_simplifier(job_type_num)


def senior_simplifier(title):
    if title == "Senior":
        return 1
    else:
        return 2
    
seniority_num = st.radio("Senior role?", options=["Senior", "Not Senior"])
seniority_num1 = senior_simplifier(seniority_num)

len_desc = st.number_input('Character Length of the Job Description', step=1.0)

st.subheader('Your skills:')
python_yn = st.radio("Python (0 for No, 1 for Yes)", options=[0, 1])
#r_yn = st.radio("R (0 for No, 1 for Yes)", options=[0, 1])
aws = st.radio("AWS (0 for No, 1 for Yes)", options=[0, 1])
#spark = st.radio("Spark (0 for No, 1 for Yes)", options=[0, 1])
excel = st.radio("Hadoop (0 for No, 1 for Yes)", options=[0, 1])
#docker_yn = st.radio("Docker (0 for No, 1 for Yes)", options=[0, 1])
#sql_yn = st.radio("SQL (0 for No, 1 for Yes)", options=[0, 1])
#linux_yn = st.radio("Linux (0 for No, 1 for Yes)", options=[0, 1])
#flask_yn = st.radio("Flask (0 for No, 1 for Yes)", options=[0, 1])
#django_yn = st.radio("Django (0 for No, 1 for Yes)", options=[0, 1])
excel = st.radio("Tensorflow (0 for No, 1 for Yes)", options=[0, 1])
#keras_yn = st.radio("Keras (0 for No, 1 for Yes)", options=[0, 1])
#pytorch_yn = st.radio("PyTorch (0 for No, 1 for Yes)", options=[0, 1])
#tableau_yn = st.radio("Tableau (0 for No, 1 for Yes)", options=[0, 1])
#algo_yn = st.radio(
    #"Strong Algorithmic Knowledge (0 for No, 1 for Yes)", options=[0, 1])
#stats_yn = st.radio(
    #"Strong Statistical Knowledge (0 for No, 1 for Yes)", options=[0, 1])
    
features = [avg_salary,Rating,Size,Type of ownership,Industry,Sector,Revenue,num_comp,hourly,employer_provided,job_state,
            same_state,age,python_yn,spark,aws,excel,job_simp,seniority,desc_len]
final_features = np.array(features).reshape(1, -1)

if st.button('Predict'):
    model = load_models()
    prediction = model.predict(final_features)
    st.balloons()
    st.success(f'Your predicted salary is US$ {round(prediction[0],3)*1000} ')
