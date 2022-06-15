import streamlit as st

caste = ['OC', 'BC-A', 'BC-B', 'BC-C', 'BC-D', 'BC-E', 'SC', 'ST']
gender = ['m', 'f']

from keras.models import load_model
model = load_model('branchpred.h5')


labels = ['Civil Engineering', 'Computer Science', 'Electronics and Communication', 'Electrical and Electronics Engineering', 'Mechanical Engineering']


st.title("Branch Predictor")
caste_ = st.selectbox("Caste", options=caste)
gender_ = st.radio("Gender", options=("Male", "Female"))
rank_ = st.number_input("Rank", min_value=1, max_value=400000, step=1)
rank_ = (rank_-13808)/(94518-13808)
caste_ = caste.index(caste_)
caste_ = (caste_-0)/7
gender_ = 0 if gender_ == "male" else 1

predict = st.button("Predict")

if predict:
    modelInput = [caste_,rank_, gender_]
    h = model.predict([modelInput]).argmax(axis=1)[0]
    st.write(f"Branch: {labels[h]}")


    