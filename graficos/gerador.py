import matplotlib.pyplot as plt
import numpy as np

# Dados baseados na discussão e benchmarks do dataset COCO
versions = ['YOLOv3\n(2018)', 'YOLOv4\n(2020)', 'YOLOv5x\n(2020)', 'YOLOv7x\n(2022)', 'YOLOv8x\n(2023)', 'YOLOv11x\n(2024)']
map_scores = [33.0, 43.5, 50.7, 51.4, 53.9, 54.7]
# Estimativa de parâmetros (em milhões) para os modelos correspondentes ("x" ou os maiores reportados para as versões antigas)
params_m = [61.5, 64.3, 86.7, 71.3, 68.2, 53.2]

# Configurando os subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Precisão (mAP)
ax1.plot(versions, map_scores, marker='o', color='#1f77b4', linewidth=3, markersize=10)
ax1.set_title('Evolução da Precisão (mAP)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Precisão (mAP no COCO)', fontsize=12)
ax1.set_ylim(25, 65)
ax1.grid(True, linestyle='--', alpha=0.6)

# Adicionando os rótulos de mAP
for i, txt in enumerate(map_scores):
    ax1.annotate(f"{txt}", (versions[i], map_scores[i]), textcoords="offset points", xytext=(0,12), ha='center', fontweight='bold')

# Gráfico 2: Eficiência Paramétrica (Milhões de Parâmetros)
bars = ax2.bar(versions, params_m, color='#ff7f0e', alpha=0.85, edgecolor='black')
ax2.set_title('Evolução do Número de Parâmetros', fontsize=14, fontweight='bold')
ax2.set_ylabel('Milhões de Parâmetros', fontsize=12)
ax2.set_ylim(0, 100)
ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

# Destacando o YOLO11 com uma cor diferente para evidenciar a eficiência
bars[-1].set_color('#2ca02c') 
bars[-1].set_edgecolor('black')

# Adicionando os rótulos de Parâmetros
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.1f}M', ha='center', va='bottom', fontweight='bold')

# Ajustando layout
plt.suptitle('Trajetória do YOLO: Precisão vs. Eficiência (2018 - 2024)', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()

# Salvando a imagem
plt.savefig('yolo_evolucao_graficos.png', dpi=300, bbox_inches='tight')