"""Problema 2 - Veículo autônomo: realçar as linhas da via (via.bmp).

Saídas:
  (a) resultados/p2_binarizada.png - imagem binarizada (bordas de Canny)
  (b) resultados/p2_rotulada.png   - imagem original com as bordas em vermelho
  extra: resultados/p2_linhas.png  - segmentos de reta da via (Hough) em vermelho
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

    # Extra: segmentos retos (linhas da pista) na metade inferior da imagem.
    h = bordas.shape[0]
    roi = bordas.copy()
    roi[: h // 2] = 0
    linhas = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50,
                             minLineLength=h // 10, maxLineGap=20)
    img_linhas = img.copy()
    if linhas is not None:
        for x1, y1, x2, y2 in linhas.reshape(-1, 4):
            cv2.line(img_linhas, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    salvar_imagem("p2_binarizada.png", bordas)
    salvar_imagem("p2_rotulada.png", rotulada)
    salvar_imagem("p2_linhas.png", img_linhas)


if __name__ == "__main__":
    main()
