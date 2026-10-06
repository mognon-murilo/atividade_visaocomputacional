"""Problema 1 - ConservaVias: contar as rodas de um veículo (car.jpg).

Saídas:
  (a) resultados/p1_rodas.png  - imagem rotulada com as rodas detectadas
  (b) resultados/p1_rodas.txt  - total de rodas e (x, y, raio) de cada círculo
"""
import cv2
import numpy as np

from util import escrever, ler_imagem, localizar_arquivo, salvar_imagem, salvar_texto


def detectar_rodas(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape

    # Rodas: círculos na metade inferior do carro, com raio proporcional à largura.
    min_r, max_r = int(w * 0.035), int(w * 0.14)
    rodas = []
    # Diminui o limiar do acumulador até encontrar pelo menos 2 rodas.
    for param2 in range(70, 19, -5):
        circulos = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w * 0.2,
                                    param1=120, param2=param2,
                                    minRadius=min_r, maxRadius=max_r)
        if circulos is None:
            continue
        rodas = [c for c in np.round(circulos[0]).astype(int) if c[1] > h * 0.45]
        if len(rodas) >= 2:
            break
    return sorted(rodas, key=lambda c: c[0])


def main():
    caminho = localizar_arquivo(["car.jpg", "car2.png"])
    print(f"Problema 1 - entrada: {caminho}")
    img = ler_imagem(caminho)
    rodas = detectar_rodas(img)

    saida = img.copy()
    linhas = [f"Arquivo: {caminho}", f"Total de rodas encontradas: {len(rodas)}", ""]
    for i, (x, y, r) in enumerate(rodas, 1):
        cv2.circle(saida, (x, y), r, (0, 255, 0), 3)
        cv2.circle(saida, (x, y), 3, (0, 0, 255), -1)
        escrever(saida, f"roda{i}", (x - r, y - r - 6))
        linhas.append(f"Roda {i}: x={x}, y={y}, raio={r}")
    escrever(saida, f"Rodas: {len(rodas)}", (10, 25), escala=0.7, espessura=2)

    salvar_imagem("p1_rodas.png", saida)
    salvar_texto("p1_rodas.txt", linhas)
    print(f"  {len(rodas)} roda(s) detectada(s)")


if __name__ == "__main__":
    main()
