import streamlit as st
import plotly.express as px
from fastai.vision.all import *
import platform
import pathlib

# WindowsPath masalasini hal qilish (Linux uchun)
plt = platform.system()
if plt == "Linux":
    pathlib.WindowsPath = pathlib.PosixPath

st.title("Transportni Classification qiluvchi model")

# Modelni kesh bilan yuklash
@st.cache_resource
def load_model():
    return load_learner("transport_model.pkl")

model = load_model()

# Rasm yuklash
file = st.file_uploader("Rasm yuklash", type=["png", "jpeg", "jpg"])
if file is not None:
    st.image(file)
    img = PILImage.create(file)

    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    fig = px.bar(x=probs * 100, y=model.dls.vocab, orientation='h')
    st.plotly_chart(fig)
