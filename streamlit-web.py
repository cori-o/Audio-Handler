import streamlit as st
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import io

st.title("오디오 품질 시각화 (Before vs After)")

# 파일 업로드
uploaded_file1 = st.file_uploader("원본 오디오 업로드", type=["wav"])
uploaded_file2 = st.file_uploader("잡음 제거 후 오디오 업로드", type=["wav"])

# 구간 슬라이더
start_sec, end_sec = st.slider("시각화할 구간 (초)", 0, 1800, (600, 1200), step=10)

if uploaded_file1 and uploaded_file2:
    # 오디오 로딩
    y1, sr1 = librosa.load(uploaded_file1, sr=None)
    y2, sr2 = librosa.load(uploaded_file2, sr=None)

    # 구간 자르기
    y1 = y1[int(start_sec * sr1):int(end_sec * sr1)]
    y2 = y2[int(start_sec * sr2):int(end_sec * sr2)]

    # 시각화
    from tempfile import NamedTemporaryFile
    from visualize_module import visualize_before_after_all  # 네 함수 import

    output_path = "compare_streamlit.png"
    visualize_before_after_all(y1, sr1, y2, sr2, file_name=output_path)

    st.image(output_path, caption="Before vs After - 시각화 결과", use_column_width=True)

    # 오디오 재생
    st.audio(uploaded_file1, format='audio/wav', start_time=start_sec)
    st.audio(uploaded_file2, format='audio/wav', start_time=start_sec)
