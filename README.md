# Visão Figurinhas

Abre a câmera e mostra, ao lado do vídeo, **apenas UMA figurinha**: o resultado
da combinação de mão(s) + rosto detectados naquele instante. Sem malha, sem
texto, sem esqueleto desenhado em cima da câmera — só o vídeo puro + a
figurinha.

**Figurinhas reais atualmente em uso** (as outras da tabela abaixo ainda são
placeholders genéricos, mostrando um "?" cinza até você colocar uma imagem
real com o mesmo nome de arquivo):

| Figurinha           | Gesto que dispara ela                          |
|----------------------|--------------------------------------------------|
| 👩 Mulher             | Pensativo (dedo perto do queixo)                 |
| 🐱 Gato de boca aberta | As duas mãos na cabeça + boca bem aberta         |
| 🐱 Gato calmo          | Rosto neutro (nenhuma expressão especial)        |

## Instalação

```bash
pip install opencv-python mediapipe numpy pillow
```

> Precisa de Python 3.9–3.11 (o MediaPipe ainda não suporta todas as versões mais novas).

## Como rodar

1. Garanta que a pasta `stickers/` tem os arquivos usados no código. As 3 já
   substituídas por imagens reais são `indicador.png`, `boca_aberta.png` e
   `rosto_neutro.png` (veja a tabela acima). Os demais nomes de arquivo ainda
   são placeholders — mostram um "?" cinza até você colocar uma imagem real
   com o mesmo nome.
2. Rode:
   ```bash
   python main.py
   ```
3. `ESC` ou `q` para sair.

## Gestos/expressões e prioridade

Só **uma** figurinha aparece por vez. Quando mais de uma coisa é detectada ao
mesmo tempo, esta é a ordem de prioridade (de cima pra baixo):

| Prioridade | Gesto / expressão               | Como fazer                                            | Arquivo             | Imagem |
|:---:|-----------------------------------|----------------------------------------------------------|------------------------|:---:|
| 1 | **Mãos na cabeça + boca aberta**     | As DUAS mãos na região da cabeça e a boca bem aberta ao mesmo tempo | `boca_aberta.png` | 🐱 real |
| 2 | Pensativo (dedo no queixo)            | Qualquer dedo de qualquer mão encostando perto do queixo  | `indicador.png`       | 👩 real |
| 3 | Mão aberta                            | Todos os 5 dedos esticados                                | `mao_aberta.png`      | placeholder |
| 3 | Punho fechado                         | Todos os dedos dobrados                                   | `punho_fechado.png`   | placeholder |
| 3 | Joia                                  | Só o polegar esticado                                      | `joia.png`            | placeholder |
| 3 | Paz e amor                            | Indicador + médio esticados                                | `paz_e_amor.png`      | placeholder |
| 3 | Indicador                             | Só o dedo indicador esticado                               | `dedo_meio.png`       | placeholder |
| 3 | Chifrinho / rock                      | Polegar + indicador + mindinho esticados                   | `chifrinho.png`       | placeholder |
| 4 | Sorrindo                              | Boca alargada (sorriso)                                    | `sorriso.png`         | placeholder |
| 5 | Rosto neutro                          | Rosto detectado, sem nenhuma expressão acima                | `rosto_neutro.png`    | 🐱 real |
| 6 | Nada detectado                        | Nem mão nem rosto na câmera                                | `nenhum.png`          | placeholder |

O item em **negrito** é o mais recente: a figurinha do gato de boca aberta agora
exige as duas mãos na região da cabeça **e** a boca bem aberta ao mesmo tempo —
não depende mais só da expressão facial sozinha.

## Como funciona por baixo dos panos

- **MediaPipe Hands** (`max_num_hands=2`) e **MediaPipe Face Mesh** rodam em
  paralelo a cada frame. Nada é desenhado na imagem — os pontos (landmarks) são
  usados só internamente para os cálculos.
- **Mãos na cabeça + boca aberta**: para cada mão, calcula o "centro da palma"
  (média do pulso + bases dos dedos) e verifica se ele está na região acima/ao
  redor da cabeça — verticalmente no nível da testa ou um pouco abaixo (para
  cobrir orelhas/topo da cabeça), e horizontalmente alinhado com a largura do
  rosto. Só considera a condição satisfeita se **as duas mãos** caírem nessa
  região ao mesmo tempo. Em paralelo, mede a abertura vertical da boca (lábio
  superior ao inferior) relativa à altura do rosto — acima de
  `LIMIAR_BOCA_ABERTA` (0.055), considera "boca aberta". A figurinha só aparece
  quando as duas condições batem juntas.
- **Dedo no queixo**: mede a distância entre a ponta de cada dedo (pontas:
  polegar, indicador, médio, anelar, mindinho) e o ponto do queixo do rosto
  (landmark 152), normalizada pela largura do rosto. Se ficar abaixo de
  `LIMIAR_QUEIXO_PROXIMIDADE` (0.5), considera "perto".
- **Gestos de mão**: `dedos_esticados()` compara a altura (Y) da ponta de cada
  dedo com a articulação do meio — ponta mais alta = dedo esticado. O polegar
  usa a posição X, porque se move diferente dos outros.
- **Sorriso**: razão entre a largura da boca (cantos) e a largura do rosto.
  Acima de `LIMIAR_SORRISO` (0.42) → sorrindo; senão → neutro.
- **Apenas um resultado por vez**: a função `escolher_resultado()` percorre a
  lista de prioridade de cima para baixo e retorna a primeira que bater. Ela
  ainda passa por uma pequena estabilização (`ESTABILIDADE_SEGUNDOS = 0.3`) pra
  não "piscar" entre gestos de um frame pro outro.

## Calibrando os limites

Esses valores no topo do `main.py` são heurísticas — ajuste-os observando sua
própria câmera/iluminação:

```python
LIMIAR_SORRISO = 0.42
LIMIAR_BOCA_ABERTA = 0.055
LIMIAR_QUEIXO_PROXIMIDADE = 0.5
MARGEM_CABECA_Y = 0.5
MARGEM_CABECA_X = 1.0
```

- Se "mãos na cabeça" não estiver reconhecendo mesmo com as mãos no lugar certo,
  **aumente** `MARGEM_CABECA_Y` e/ou `MARGEM_CABECA_X` (região mais permissiva).
  Se estiver disparando com as mãos longe da cabeça, **diminua** esses valores.
  **Importante:** quando a mão encosta de verdade em cima da cabeça, a câmera
  enxerga mais o dorso da mão (coberto pelo cabelo) do que a palma, e o
  MediaPipe pode simplesmente parar de detectar a mão (não é um problema de
  região, é a mão sumindo da detecção). Se isso acontecer:
  - Os valores `min_detection_confidence` e `min_tracking_confidence` do
    `mp_hands.Hands(...)` em `main()` já foram reduzidos (0.4 / 0.3) pra
    aceitar poses mais difíceis de enxergar — se ainda faltar, tente baixar
    mais um pouco (ex: 0.3 / 0.25).
  - Poses tipo "mãos nas laterais da cabeça / têmporas" (como o gesto de
    "mente explodindo") tendem a funcionar melhor que "mão achatada no topo
    da cabeça", porque a câmera ainda enxerga parte da palma/dedos.
- Se "boca aberta" não estiver disparando, **diminua** `LIMIAR_BOCA_ABERTA`.
- Se o "sorrindo" estiver disparando fácil demais, **aumente** `LIMIAR_SORRISO`.
- Se o "dedo no queixo" não reconhecer mesmo encostando, **aumente**
  `LIMIAR_QUEIXO_PROXIMIDADE`; se disparar mesmo com o dedo longe, diminua.

## Personalizando / adicionando novos gestos

1. Rode o programa e adicione um `print(dedos_esticados(...))` dentro do loop
   pra ver qual padrão de 0s e 1s aparece para o gesto que você quer mapear.
2. Adicione a entrada no dicionário `GESTOS_MAO` em `main.py`.
3. Coloque o arquivo `.png` correspondente na pasta `stickers/`.

## Próximos passos possíveis

- Detectar outras expressões (olhos fechados, sobrancelha levantada, língua de
  fora) usando outros pontos do Face Mesh.
- Trocar as regras geométricas por um classificador treinado (KNN/Random Forest
  do scikit-learn) usando os landmarks como features.
- Registrar um histórico de gestos detectados por sessão.
