import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# 화자별 발언 구간
segments = [[(3.99659375, 10.864718750000002), 'SPEAKER_01'], [(10.864718750000002, 16.21409375), 'SPEAKER_00'], [(16.450343750000002, 17.80034375), 'SPEAKER_01'], [(18.00284375, 21.17534375), 'SPEAKER_00'], [(20.12909375, 22.035968750000002), 'SPEAKER_01'], [(26.676593750000002, 33.46034375), 'SPEAKER_02'], [(29.00534375, 29.96721875), 'SPEAKER_01'], [(42.06659375, 44.91846875), 'SPEAKER_01'], [(44.96909375, 55.634093750000005), 'SPEAKER_00'], [(54.62159375, 56.41034375), 'SPEAKER_01'], [(89.89034375, 91.32471875), 'SPEAKER_03'], [(110.78159375000001, 113.97096875000001), 'SPEAKER_00'], [(115.64159375000001, 120.09659375000001), 'SPEAKER_01'], [(120.48471875000001, 121.76721875000001), 'SPEAKER_00'], [(122.59409375000001, 128.29784375), 'SPEAKER_01'], [(127.92659375000001, 135.89159375), 'SPEAKER_00'], [(135.84096875, 138.84471875), 'SPEAKER_02'], [(136.65096875, 138.74346875), 'SPEAKER_00'], [(139.24971875, 144.97034375), 'SPEAKER_00'], [(145.03784375, 145.79721875), 'SPEAKER_02'], [(146.28659375, 149.61096875), 'SPEAKER_02'], [(151.18034375000002, 154.52159375000002), 'SPEAKER_00'], [(153.32346875000002, 155.80409375000002), 'SPEAKER_02'], [(154.62284375000002, 156.07409375), 'SPEAKER_00'], [(156.25971875000002, 157.55909375000002), 'SPEAKER_00'], [(157.96409375000002, 162.65534375000001), 'SPEAKER_02'], [(162.70596875, 169.82721875000001), 'SPEAKER_02'], [(170.75534375, 175.12596875), 'SPEAKER_00'], [(175.12596875, 176.03721875000002), 'SPEAKER_02'], [(176.03721875000002, 176.74596875), 'SPEAKER_00'], [(176.94846875000002, 178.55159375000002), 'SPEAKER_00']]
speaker_colors = {
    'SPEAKER_00': '#e6194b',  # 빨강
    'SPEAKER_01': '#4363d8',  # 파랑
    'SPEAKER_02': '#ffe119',  # 노랑
    'SPEAKER_03': '#90EE90',  # 연두
}

# 원하는 위 → 아래 순서
speaker_order = ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02', 'SPEAKER_03']

# y축 위치를 위에서 아래로 매핑
y_pos_map = {speaker: idx for idx, speaker in enumerate(speaker_order[::-1])}

fig, ax = plt.subplots(figsize=(15, 4))

# 그리기
for (start, end), speaker in segments:
    y = y_pos_map[speaker]
    ax.broken_barh([(start, end - start)], (y - 0.2, 0.4), facecolors=speaker_colors[speaker])

# Y축 위치 & 라벨 매칭
yticks = list(range(len(speaker_order)))
yticklabels = speaker_order  # 위 → 아래 순서 유지

ax.set_yticks(yticks)
ax.set_yticklabels(speaker_order[::-1])  # y=0이 아래이므로 순서를 뒤집어 줌
ax.set_xlabel("Time (s)")
ax.set_title("Speaker Diarization Timeline")
ax.grid(True, axis='x', linestyle='--', alpha=0.2)

# 범례
patches = [mpatches.Patch(color=color, label=speaker) for speaker, color in speaker_colors.items()]
ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.show()