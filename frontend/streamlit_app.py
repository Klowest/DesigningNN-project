import streamlit as st
import requests
from requests.exceptions import ConnectionError
import os
import base64

IP_API = "detection-api"
PORT_API = "5000"
API_URL = f"http://{IP_API}:{PORT_API}/process"

# Путь к фоновому изображению
BACKGROUND_PATH = "./assets/background.jpg"

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Установка фона и стилей
def set_background_and_style(png_file):
    bin_str = ""
    mime = "image/jpeg"
    if os.path.exists(png_file):
        bin_str = get_base64_of_bin_file(png_file)
        ext = png_file.split('.')[-1].lower()
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
    else:
        st.warning(f"Фон не найден: {png_file}")

    st.markdown(f"""
    <style>
        /* Фоновое изображение */
        .stApp {{
            background-image: url("data:{mime};base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Основной контейнер с контентом — белая карточка */
        .main .block-container {{
            background: rgba(255, 255, 255, 0.94);
            border-radius: 20px;
            padding: 2.5rem !important;
            max-width: 850px;
            margin: 2rem auto;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
            backdrop-filter: blur(4px); /* эффект стекла (работает не во всех браузерах) */
            border: 1px solid rgba(255, 255, 255, 0.7);
        }}

        /* Заголовок */
        h1 {{
            text-align: center;
            color: #1e293b;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        .stCaption {{
            text-align: center;
            color: #64748b;
            margin-bottom: 2rem;
        }}

        /* Кнопки */
        .stButton > button {{
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-size: 1.05rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }}
        .stButton > button:hover {{
            background: linear-gradient(135deg, #2563eb, #1e40af);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
        }}
        .stButton > button:active {{
            transform: translateY(0);
        }}

        /* Загрузчики */
        [data-testid="stFileUploader"] {{
            margin-bottom: 1.5rem;
        }}

        /* Превью медиа */
        .uploaded-media {{
            text-align: center;
            margin: 1.2rem 0;
        }}
        .uploaded-media img, .uploaded-media video {{
            max-height: 280px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin: 0 auto;
            display: block;
        }}

        /* Подвал */
        .footer {{
            text-align: center;
            font-size: 0.9em;
            color: #94a3b8;
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e2e8f0;
        }}

        /* Обеспечиваем читаемость текста на фоне */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {{
            background-color: white !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
        }}

        /* Спиннер и уведомления — поверх фона */
        .stSpinner > div {{
            color: #1e40af !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# Установка фона
if os.path.exists(BACKGROUND_PATH):
    set_background_and_style(BACKGROUND_PATH)
else:
    st.warning("Фоновое изображение не найдено. Путь: " + BACKGROUND_PATH)

st.title("🎭 Детекция на фото и видео")
st.caption("Загрузите изображение или видео - мы его обработаем и вернём")

# Инициализация состояния
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.file_type = None
    st.session_state.result_bytes = None
    st.session_state.result_name = None

# Загрузка файлов
col1, col2 = st.columns(2)
with col1:
    image_file = st.file_uploader("📷 Изображение", type=["png", "jpg", "jpeg"], key="image")
    if image_file:
        st.session_state.uploaded_file = image_file
        st.session_state.file_type = "image"
with col2:
    video_file = st.file_uploader("🎥 Видео", type=["mp4", "avi", "mov", "mkv"], key="video")
    if video_file:
        st.session_state.uploaded_file = video_file
        st.session_state.file_type = "video"

# Мини-превью
if st.session_state.uploaded_file:
    st.markdown("<div class='uploaded-media'>", unsafe_allow_html=True)
    if st.session_state.file_type == "image":
        st.image(st.session_state.uploaded_file, use_container_width=True)
    elif st.session_state.file_type == "video":
        st.video(st.session_state.uploaded_file)
    st.markdown("</div>", unsafe_allow_html=True)

# Кнопка отправки
if st.button("🚀 Отправить на обработку", type="primary"):
    file = st.session_state.uploaded_file
    if not file:
        st.warning("❗ Сначала загрузите файл")
    else:
        try:
            files = {"file": (file.name, file.getvalue(), file.type)}
            with st.spinner("⏳ Обработка..."):
                response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                st.session_state.result_bytes = response.content
                # Имя файла берём из Content-Disposition или придумываем
                content_disposition = response.headers.get("content-disposition")
                if content_disposition and "filename=" in content_disposition:
                    fname = content_disposition.split("filename=")[1].strip('"')
                else:
                    ext = os.path.splitext(file.name)[1]
                    fname = f"processed_{file.name}"
                st.session_state.result_name = fname
                st.success("✅ Файл успешно обработан!")
            else:
                st.error(f"❌ Ошибка сервера: {response.status_code}")
        except ConnectionError:
            st.error("🔌 Не удаётся подключиться к серверу `detection-api`")
        except Exception as e:
            st.error(f"💥 Неожиданная ошибка: {str(e)}")

# Кнопка скачивания результата
if st.session_state.result_bytes and st.session_state.result_name:
    st.download_button(
        label="💾 Скачать обработанный файл",
        data=st.session_state.result_bytes,
        file_name=st.session_state.result_name,
        mime="application/octet-stream",
        use_container_width=True
    )

st.markdown("<div class='footer'>© 2025 Детекция объектов | Streamlit + FastAPI</div>", unsafe_allow_html=True)