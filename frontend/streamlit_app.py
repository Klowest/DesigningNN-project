import streamlit as st
import requests
from requests.exceptions import ConnectionError

ip_api = "detection-api"
port_api = "5000" 

st.title("Детекция на видео")

input_text = st.text_area(
    label="ПРИВИТ",
    height=200,
    placeholder="оу май, тут что-то можно писать",
    key="input_text"
)

if st.button("Кнопочка", use_container_width=True):
    if not input_text:
        st.warning("Пожалуйста, введите что-нибудь сначала")
    else:
        try:
            response = requests.post(
                f"http://{ip_api}:{port_api}/test",
                json={"text": input_text},  #
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.output_result = result["text"]
            else:
                st.session_state.output_result = f"Ошибка ({response.status_code})"
            
        except ConnectionError:
            st.error("Ошибка подключения к серверу")

# Результат перевода (елси вышла какая либо ошикба, то она выводится в этом поле)
output_text = st.session_state.get("output_result", "") 

st.text_area(
    label="ВЫХОД",
    value=output_text,
    height=200,
    disabled=True,
    key="output_text"
)