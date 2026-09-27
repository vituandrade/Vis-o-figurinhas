"""
Projeto: Visao Figurinhas
Detecta as maos e o rosto e mostra, ao lado da camera, UMA UNICA figurinha
resultado (sem malha, sem texto, sem overlays na imagem da camera).

Gestos/expressoes atualmente mapeados:
  - As DUAS maos na cabeca + boca bem aberta  -> boca_aberta.png
  - Dedo perto do queixo (qualquer mao)       -> indicador.png   ("Pensativo")
  - So o dedo indicador esticado               -> dedo_meio.png   ("Indicador")

Prioridade quando mais de uma coisa e detectada ao mesmo tempo (ordem de cima
para baixo na lista acima): maos na cabeca + boca aberta > dedo perto do
queixo > gesto de mao > sorriso > rosto neutro > nada.

Controles:
  - ESC ou 'q' : sai do programa
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

STICKERS_DIR = os.path.join(os.path.dirname(__file__), "stickers")

# Gestos de mao reconhecidos por padrao de dedos esticados
# (polegar, indicador, medio, anelar, mindinho) -> (nome, arquivo)
GESTOS_MAO = {
    (1, 1, 1, 1, 1): ("Mao aberta", "mao_aberta.png"),
    (0, 0, 0, 0, 0): ("Punho fechado", "punho_fechado.png"),
    (1, 0, 0, 0, 0): ("Joia", "joia.png"),
    (0, 1, 1, 0, 0): ("Paz e amor", "paz_e_amor.png"),
    (0, 1, 0, 0, 0): ("Indicador", "dedo_meio.png"),
    (1, 1, 0, 0, 1): ("Chifrinho / rock", "chifrinho.png"),
}

STICKER_PADRAO = "nenhum.png"

# --- limites (heuristicas) — ajuste testando na sua propria camera/iluminacao ---
LIMIAR_SORRISO = 0.42            # largura da boca / largura do rosto -> acima disso, sorriso
LIMIAR_BOCA_ABERTA = 0.055       # abertura vertical da boca / altura do rosto -> acima disso, boca aberta
LIMIAR_QUEIXO_PROXIMIDADE = 0.5  # distancia dedo-queixo / largura do rosto -> abaixo disso, "perto do queixo"
MARGEM_CABECA_Y = 0.5            # o quanto abaixo da testa ainda conta como "na cabeca" (fracao da altura do rosto)
MARGEM_CABECA_X = 1.0            # o quanto para os lados da bochecha ainda conta como "na cabeca" (fracao da largura do rosto)

STICKER_SIZE = 320
PAINEL_LARGURA = STICKER_SIZE + 40


def carregar_stickers():
    arquivos = {arquivo for _, arquivo in GESTOS_MAO.values()}
    arquivos.update({STICKER_PADRAO, "sorriso.png", "rosto_neutro.png", "boca_aberta.png", "indicador.png"})
    return {nome: _carregar_imagem(nome) for nome in arquivos}


def _carregar_imagem(nome_arquivo):
    caminho = os.path.join(STICKERS_DIR, nome_arquivo)
    img = cv2.imread(caminho, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = np.full((STICKER_SIZE, STICKER_SIZE, 3), 60, dtype=np.uint8)
        cv2.putText(img, "?", (STICKER_SIZE // 2 - 30, STICKER_SIZE // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        return img
    return cv2.resize(img, (STICKER_SIZE, STICKER_SIZE))


def dedos_esticados(hand_landmarks, mao_label):
    """(polegar, indicador, medio, anelar, mindinho): 1 = esticado, 0 = dobrado."""
    lm = hand_landmarks.landmark
    dedos = []
    if mao_label == "Right":
        dedos.append(1 if lm[4].x < lm[3].x else 0)
    else:
        dedos.append(1 if lm[4].x > lm[3].x else 0)

    pontas = [8, 12, 16, 20]
    juntas = [6, 10, 14, 18]
    for ponta, junta in zip(pontas, juntas):
        dedos.append(1 if lm[ponta].y < lm[junta].y else 0)
    return tuple(dedos)


def classificar_boca(face_landmarks, w_img, h_img):
    """Retorna (nome, arquivo) da expressao da boca: Sorrindo / Neutro."""
    lm = face_landmarks.landmark

    def ponto(i):
        return np.array([lm[i].x * w_img, lm[i].y * h_img])

    largura_boca = np.linalg.norm(ponto(291) - ponto(61))
    largura_rosto = np.linalg.norm(ponto(454) - ponto(234))
    if largura_rosto == 0:
        return ("Neutro", "rosto_neutro.png")

    razao = largura_boca / largura_rosto
    if razao >= LIMIAR_SORRISO:
        return ("Sorrindo", "sorriso.png")
    return ("Neutro", "rosto_neutro.png")


def boca_esta_aberta(face_landmarks, w_img, h_img):
    """True se a boca estiver bem aberta (abertura vertical grande em relacao ao rosto)."""
    lm = face_landmarks.landmark

    def ponto(i):
        return np.array([lm[i].x * w_img, lm[i].y * h_img])

    labio_sup, labio_inf = ponto(13), ponto(14)
    testa, queixo = ponto(10), ponto(152)
    altura_rosto = np.linalg.norm(queixo - testa)
    if altura_rosto == 0:
        return False
    abertura = np.linalg.norm(labio_inf - labio_sup) / altura_rosto
    return abertura >= LIMIAR_BOCA_ABERTA


def dedo_perto_do_queixo(hand_landmarks, face_landmarks, w_img, h_img):
    """True se alguma ponta de dedo dessa mao estiver perto do queixo do rosto."""
    queixo = np.array([face_landmarks.landmark[152].x * w_img, face_landmarks.landmark[152].y * h_img])
    largura_rosto = np.linalg.norm(
        np.array([face_landmarks.landmark[454].x * w_img, face_landmarks.landmark[454].y * h_img])
        - np.array([face_landmarks.landmark[234].x * w_img, face_landmarks.landmark[234].y * h_img])
    )
    if largura_rosto == 0:
        return False

    limite = LIMIAR_QUEIXO_PROXIMIDADE * largura_rosto
    for idx_ponta in (4, 8, 12, 16, 20):  # pontas de todos os dedos
        lm = hand_landmarks.landmark[idx_ponta]
        ponta = np.array([lm.x * w_img, lm.y * h_img])
        if np.linalg.norm(ponta - queixo) < limite:
            return True
    return False


def centro_da_palma(hand_landmarks, w_img, h_img):
    """Aproxima o centro da palma pela media das bases dos dedos (mais estavel que so o pulso)."""
    idxs = (0, 5, 9, 13, 17)  # pulso + bases dos 4 dedos (exceto polegar) + base do indicador
    pontos = [
        np.array([hand_landmarks.landmark[i].x * w_img, hand_landmarks.landmark[i].y * h_img])
        for i in idxs
    ]
    return np.mean(pontos, axis=0)


def mao_esta_na_cabeca(centro_palma, face_landmarks, w_img, h_img):
    """
    True se o centro da palma estiver na regiao acima/ao redor da cabeca:
    verticalmente no nivel da testa (ou acima) e horizontalmente alinhado com a largura do rosto.
    """
    lm = face_landmarks.landmark

    def ponto(i):
        return np.array([lm[i].x * w_img, lm[i].y * h_img])

    testa, queixo = ponto(10), ponto(152)
    bochecha_esq, bochecha_dir = ponto(234), ponto(454)

    altura_rosto = np.linalg.norm(queixo - testa)
    largura_rosto = np.linalg.norm(bochecha_dir - bochecha_esq)
    if altura_rosto == 0 or largura_rosto == 0:
        return False

    limite_y = testa[1] + MARGEM_CABECA_Y * altura_rosto
    limite_x_min = bochecha_esq[0] - MARGEM_CABECA_X * largura_rosto
    limite_x_max = bochecha_dir[0] + MARGEM_CABECA_X * largura_rosto

    dentro_y = centro_palma[1] < limite_y
    dentro_x = limite_x_min < centro_palma[0] < limite_x_max
    return dentro_y and dentro_x


def maos_na_cabeca_e_boca_aberta(hand_results, face_landmarks, w_img, h_img):
    """True se as DUAS maos estiverem na regiao da cabeca E a boca estiver bem aberta."""
    if face_landmarks is None:
        return False
    maos = hand_results.multi_hand_landmarks or []
    if len(maos) < 2:
        return False

    maos_na_cabeca = sum(
        1 for hand_landmarks in maos
        if mao_esta_na_cabeca(centro_da_palma(hand_landmarks, w_img, h_img), face_landmarks, w_img, h_img)
    )
    if maos_na_cabeca < 2:
        return False

    return boca_esta_aberta(face_landmarks, w_img, h_img)


def escolher_resultado(hand_results, face_results, w_img, h_img):
    """
    Decide qual UNICA figurinha mostrar neste frame, seguindo a ordem de prioridade:
    maos na cabeca + boca aberta > dedo perto do queixo > gesto de mao > sorrindo >
    rosto neutro > nada.
    """
    maos = hand_results.multi_hand_landmarks or []
    handedness = hand_results.multi_handedness or []
    rosto = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None

    # 1) as duas maos na cabeca + boca bem aberta
    if maos_na_cabeca_e_boca_aberta(hand_results, rosto, w_img, h_img):
        return ("Maos na cabeca + boca aberta", "boca_aberta.png")

    # 2) dedo perto do queixo (precisa de rosto + pelo menos uma mao)
    if rosto is not None:
        for hand_landmarks in maos:
            if dedo_perto_do_queixo(hand_landmarks, rosto, w_img, h_img):
                return ("Pensativo (dedo no queixo)", "indicador.png")

    # 3) gesto de mao reconhecido (primeira mao que bater com algum padrao)
    for hand_landmarks, hd in zip(maos, handedness):
        mao_label = hd.classification[0].label
        padrao = dedos_esticados(hand_landmarks, mao_label)
        if padrao in GESTOS_MAO:
            return GESTOS_MAO[padrao]

    # 4) e 5) expressao do rosto (sorrindo) ou neutro
    if rosto is not None:
        return classificar_boca(rosto, w_img, h_img)

    # 6) nada detectado
    return ("Nada detectado", STICKER_PADRAO)


def sobrepor_com_alpha(fundo, sticker, x, y):
    h, w = sticker.shape[:2]
    if sticker.shape[2] == 4:
        alpha = sticker[:, :, 3] / 255.0
        for c in range(3):
            fundo[y:y + h, x:x + w, c] = (
                alpha * sticker[:, :, c] + (1 - alpha) * fundo[y:y + h, x:x + w, c]
            )
    else:
        fundo[y:y + h, x:x + w] = sticker[:, :, :3]
    return fundo


def main():
    if not os.path.isdir(STICKERS_DIR) or not os.listdir(STICKERS_DIR):
        print("Pasta de figurinhas vazia ou ausente. Rode 'python gerar_stickers.py' primeiro.")
        return

    stickers = carregar_stickers()

    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nao foi possivel abrir a camera.")
        return

    # estabilizacao: evita a figurinha "piscando" entre frames
    ESTABILIDADE_SEGUNDOS = 0.3
    arquivo_atual = STICKER_PADRAO
    arquivo_estavel = STICKER_PADRAO
    desde = 0.0

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.3,
    ) as hands, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_img, w_img = frame.shape[:2]

            resultado_maos = hands.process(rgb)
            resultado_rosto = face_mesh.process(rgb)

            _nome, arquivo_bruto = escolher_resultado(resultado_maos, resultado_rosto, w_img, h_img)

            agora = time.time()
            if arquivo_bruto != arquivo_atual:
                arquivo_atual = arquivo_bruto
                desde = agora
            if agora - desde > ESTABILIDADE_SEGUNDOS:
                arquivo_estavel = arquivo_atual

            # painel lateral com APENAS a figurinha resultado, sem texto/overlay
            painel = np.full((h_img, PAINEL_LARGURA, 3), 30, dtype=np.uint8)
            y_offset = max((h_img - STICKER_SIZE) // 2, 0)
            sobrepor_com_alpha(painel, stickers[arquivo_estavel], 20, y_offset)

            saida = np.hstack([frame, painel])
            cv2.imshow("Visao Figurinhas (ESC para sair)", saida)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27 or tecla == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()