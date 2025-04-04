import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# 화자별 발언 구간
segments = [[(4.097843750000001, 10.94909375), 'SPEAKER_01'], [(10.88159375, 16.24784375), 'SPEAKER_00'], [(16.46721875, 17.867843750000002), 'SPEAKER_01'], [(17.96909375, 19.94346875), 'SPEAKER_00'], [(20.11221875, 22.01909375), 'SPEAKER_01'], [(20.14596875, 21.15846875), 'SPEAKER_00'], [(26.676593750000002, 27.773468750000003), 'SPEAKER_01'], [(29.039093750000003, 31.722218750000003), 'SPEAKER_01'], [(37.814093750000005, 38.62409375), 'SPEAKER_00'], [(42.06659375, 44.71596875), 'SPEAKER_01'], [(44.95221875, 48.057218750000004), 'SPEAKER_00'], [(48.057218750000004, 49.28909375), 'SPEAKER_01'], [(49.525343750000005, 50.65596875), 'SPEAKER_00'], [(51.38159375, 54.82409375), 'SPEAKER_00'], [(54.570968750000006, 55.38096875), 'SPEAKER_01'], [(85.63784375, 86.39721875000001), 'SPEAKER_01'], [(87.12284375, 91.42596875000001), 'SPEAKER_01'], [(110.83221875000001, 113.93721875000001), 'SPEAKER_00'], [(115.65846875000001, 120.09659375000001), 'SPEAKER_01'], [(120.48471875000001, 121.75034375000001), 'SPEAKER_00'], [(122.59409375000001, 126.13784375), 'SPEAKER_01'], [(126.71159375, 128.21346875), 'SPEAKER_01'], [(127.53846875, 130.50846875000002), 'SPEAKER_00'], [(130.86284375, 135.82409375), 'SPEAKER_00'], [(135.87471875, 138.79409375), 'SPEAKER_01'], [(139.58721875, 142.72596875000002), 'SPEAKER_00'], [(142.99596875, 144.95346875), 'SPEAKER_00'], [(146.33721875, 148.46346875), 'SPEAKER_01'], [(148.69971875000002, 149.52659375000002), 'SPEAKER_01'], [(151.19721875000002, 153.64409375), 'SPEAKER_00'], [(153.30659375000002, 155.06159375000001), 'SPEAKER_01'], [(154.79159375, 155.77034375000002), 'SPEAKER_00'], [(156.41159375, 157.50846875000002), 'SPEAKER_00'], [(157.96409375000002, 159.56721875000002), 'SPEAKER_01'], [(159.78659375, 162.38534375), 'SPEAKER_01'], [(162.92534375, 165.89534375000002), 'SPEAKER_01'], [(166.19909375, 167.51534375), 'SPEAKER_01'], [(170.80596875, 172.15596875), 'SPEAKER_00'], [(172.52721875, 173.38784375), 'SPEAKER_00'], [(173.60721875000002, 175.12596875), 'SPEAKER_00'], [(175.12596875, 175.96971875), 'SPEAKER_01'], [(177.13409375, 178.50096875), 'SPEAKER_00']]
speaker_colors = {
    'SPEAKER_00': '#e6194b',  # 빨강
    'SPEAKER_01': '#4363d8',  # 파랑
}

# 원하는 위 → 아래 순서
speaker_order = ['SPEAKER_00', 'SPEAKER_01']

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