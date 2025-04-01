import streamlit as st
import librosa
import matplotlib.pyplot as plt
from src import AudioVisualizer
import librosa.display
import numpy as np
import io

st.markdown(
    """
    <h1 style='text-align: center; white-space: nowrap; font-size: 32px; margin-bottom: 16px'>
        오디오 품질 시각화 (Before vs After)
    </h1>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <h5 style='text-align: center; margin-bottom: 4px;'>
            🎧 원본 오디오 업로드
        </h5>
        """, unsafe_allow_html=True
    )
    uploaded_file1 = st.file_uploader("", type=["wav"], key="before", label_visibility="collapsed")

with col2:
    st.markdown(
        """
        <h5 style='text-align: center; margin-bottom: 4px;'>
            🔇 잡음 제거 후 오디오 업로드
        </h5>
        """, unsafe_allow_html=True
    )
    uploaded_file2 = st.file_uploader("", type=["wav"], key="after", label_visibility="collapsed")

# ✔️ 자분 제거 후 파일 길이 기준으로 그래프 구간 설정
if uploaded_file2 is not None:
    audio_bytes2 = uploaded_file2.read()
    buffer2 = io.BytesIO(audio_bytes2)
    y2_full, sr2 = librosa.load(buffer2, sr=None)

    duration_sec = int(len(y2_full) / sr2)
    print(duration_sec)

    start_sec, end_sec = st.slider(
        "시각화할 구간 (초)",
        min_value=0,
        max_value=duration_sec,
        value=(0, min(60, duration_sec)),
        step=1
    )
    print(start_sec, end_sec)

# ✔️ 연결 가능 하면 구간 자분 + 시각화 진행
if uploaded_file1 and uploaded_file2:
    # 원본 파일 로드
    buffer1 = io.BytesIO(uploaded_file1.getvalue())
    buffer2 = io.BytesIO(uploaded_file2.getvalue())
    y1, sr1 = librosa.load(buffer1, sr=None)
    y2, sr2 = librosa.load(buffer2, sr=None)

    # 구간 자분
    y1 = y1[int(start_sec * sr1):int(end_sec * sr1)]
    y2 = y2[int(start_sec * sr2):int(end_sec * sr2)]

    # 시각화 진행
    audio_visualizer = AudioVisualizer()
    output_path = "compare_streamlit.png"
    audio_visualizer.visualize_before_after_all(y1, sr1, y2, sr2, file_name=output_path)

    st.image(output_path, caption="Before vs After - 시각화 결과", use_column_width=True)

    # 오디오 재생 (start_time 메가변수인지는 확인)
    st.audio(uploaded_file1, format='audio/wav')
    st.audio(uploaded_file2, format='audio/wav')
