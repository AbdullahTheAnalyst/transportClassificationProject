import streamlit as st
import plotly.express as px
from fastai.vision.all import *
import platform
import pathlib

plt = platform.system()
if plt == "Linux":
  pathlib.WindowsPath = pathlib.PosixPath


# title
st.title("Transportni Classification qiluvchi model")

# rasmni joylash
file = st.file_uploader("Rasm yuklash", 
                 type = ["png", "jpeg", "gif", "svg"])
st.image(file)
# PIL convert qilish, agar mavjud bo'lsa
if file is not None:
    img = PILImage.create(file)

# modelni yuklash fastai orqali
model = load_learner("transport_model.pkl")

# prediction
pred, pred_id, probs = model.predict(img)
st.success(f"Bashorat: {pred}")
st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

# plotitng
figure = px.bar(x = probs * 100, y = model.dls.vocab)
st.plotly_chart(figure)
