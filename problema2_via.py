"""Problema 2 - Veículo autônomo: realçar as linhas da via (via.bmp).

Saídas:
  (a) resultados/p2_binarizada.png - imagem binarizada (bordas de Canny)
  (b) resultados/p2_rotulada.png   - imagem original com as bordas em vermelho
  extra: resultados/p2_linhas.png  - faixas pintadas da via (Hough) em vermelho
"""
import cv2
import numpy as np

from util import ler_imagem, localizar_arquivo, salvar_imagem


def main():
    caminho = localizar_arquivo(["via.bmp"])
    print(f"Problema 2 - entrada: {caminho}")
    img = ler_imagem(caminho)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gray, (5, 5), 0)
    # Limiares do Canny calculados a partir da mediana (funciona em imagens claras/escuras).
    mediana = np.median(suave)
    baixo, alto = int(max(0, 0.66 * mediana)), int(min(255, 1.33 * mediana))
    bordas = cv2.Canny(suave, baixo, alto)

    rotulada = img.copy()
    rotulada[bordas > 0] = (0, 0, 255)

    # Extra: realce só da tinta da pista (branca ou amarela saturada) que encosta no
    # asfalto escuro, seguido de Canny + HoughLinesP para obter os segmentos das faixas.
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    branco = cv2.inRange(hsv, (0, 0, 170), (179, 50, 255))
    amarelo = cv2.inRange(hsv, (12, 110, 120), (35, 255, 255))
    asfalto = cv2.inRange(hsv, (0, 0, 0), (179, 255, 60))  # asfalto: muito escuro
    perto_asfalto = cv2.dilate(asfalto, np.ones((9, 9), np.uint8))
    tinta = (branco | amarelo) & perto_asfalto
    h = tinta.shape[0]
    tinta[: int(h * 0.4)] = 0
    linhas = cv2.HoughLinesP(cv2.Canny(tinta, 50, 150), 1, np.pi / 180, threshold=30,
                             minLineLength=h // 15, maxLineGap=25)
    img_linhas = img.copy()
    if linhas is not None:
        for x1, y1, x2, y2 in linhas.reshape(-1, 4):
            if abs(y2 - y1) < 0.2 * abs(x2 - x1):
                continue  # descarta segmentos quase horizontais
            cv2.line(img_linhas, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    salvar_imagem("p2_binarizada.png", bordas)
    salvar_imagem("p2_rotulada.png", rotulada)
    salvar_imagem("p2_linhas.png", img_linhas)


if __name__ == "__main__":
    main()
