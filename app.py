import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("/content/drive/My Drive/Colab Notebooks/NSP_Creditcardfrauddetection.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('/content/drive/My Drive/Colab Notebooks/Dataset/card_transdata.csv')
X = dataset.iloc[:, [0:8]].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin_number, online_order):
  output= model.predict(sc.transform([[distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin_number, online_order]]))
  print("Fraud", output)
  if output==[1]:
    prediction="It is a Fraud"
  else:
    prediction="It is not a Fraud"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Group Of Institution</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Artificial Intelligence and Data Science</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">ML_Lab Project Deployment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Credi Card Fraud Detection")
    distance_from_home = st.number_input("Distance From Home","")
    distance_from_last_transaction = st.number_input("Distance From Last Transaction") 
    ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price")
    repeat_retailer = st.number_input("Repeat Retailer")
    used_chip = st.number_input("Used Chip")
    used_pin_number = st.number_input("Used Pin Number")
    online_order = st.number_input("Online Order")

    result=""
    if st.button("Predict"):
      result=predict_note_authentication(distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin_number, online_order)
      st.success('Model has detected {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Team(Alok Raj*, Aditya Burman, Mahi Ajay")
      st.subheader("Student , Department Artificial Intelligence and Data Science")

if __name__=='__main__':
  main()
