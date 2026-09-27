"""
Gera figurinhas placeholder simples (emoji-like) para cada gesto reconhecido.
Rode este script uma vez para popular a pasta 'stickers/'.
Depois, é só substituir esses arquivos pelos .png/.webp das suas figurinhas reais do WhatsApp,
mantendo os mesmos nomes de arquivo.
"""
import cv2
import numpy as np
import os

STICKERS_DIR = os.path.join(os.path.dirname(__file__), "stickers")
os.makedirs(STICKERS_DIR, exist_ok=True)

# (nome_do_arquivo, texto_curto, cor_de_fundo BGR)
# OpenCV não desenha emojis de verdade, então usamos siglas/texto como placeholder.
# Troque os arquivos depois pelas figurinhas reais (mesmo nome de arquivo).
STICKERS = [
    ("mao_aberta.png", "MAO", (80, 160, 255)),
    ("joia.png", "OK", (80, 220, 80)),
    ("paz_e_amor.png", "PAZ", (255, 180, 80)),
    ("punho_fechado.png", "SOCO", (100, 100, 220)),
    ("chifrinho.png", "ROCK", (200, 80, 200)),
    ("indicador.png", "OI", (80, 200, 220)),
    ("nenhum.png", "...", (120, 120, 120)),
    # figurinhas de rosto
    ("sorriso.png", ":)", (255, 210, 80)),
    ("rosto_neutro.png", ":|", (150, 150, 150)),
]

SIZE = 300

for filename, text, color in STICKERS:
    img = np.full((SIZE, SIZE, 4), 0, dtype=np.uint8)  # canal alpha
    # círculo de fundo colorido
    cv2.circle(img, (SIZE // 2, SIZE // 2), SIZE // 2 - 10, (*color, 255), -1)
    cv2.circle(img, (SIZE // 2, SIZE // 2), SIZE // 2 - 10, (255, 255, 255, 255), 6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3 if len(text) <= 3 else 1.0
    thickness = 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    org = ((SIZE - tw) // 2, (SIZE + th) // 2)
    cv2.putText(img, text, org, font, scale, (255, 255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imwrite(os.path.join(STICKERS_DIR, filename), img)

print(f"Figurinhas placeholder geradas em: {STICKERS_DIR}")
print("Troque esses arquivos pelas suas figurinhas reais (mesmo nome, fundo transparente em PNG).")