"""Problema 1 - ConservaVias: contar as rodas de um veículo (car.jpg).

Saídas:
  (a) resultados/p1_<imagem>_rodas.png - imagem rotulada com as rodas detectadas
  (b) resultados/p1_<imagem>_rodas.txt - total de rodas e (x, y, raio) de cada círculo

Sem argumentos processa car.jpg e car2.png.
"""
import os

import cv2
import numpy as np

from util import (arquivos_entrada, escrever, ler_imagem, nome_base, salvar_imagem,
                  salvar_texto)


def regiao_do_carro(gray):
    """Retângulo (x, y, w, h) do veículo: maior objeto que se destaca do fundo claro."""
    _, mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(maior) > 0.1 * h * w:
            return cv2.boundingRect(maior)
    return 0, 0, w, h  # fundo não é claro: usa a imagem inteira


def detectar_rodas(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cx, cy, cw, ch = regiao_do_carro(gray)
    suave = cv2.medianBlur(gray, 5)

    # Procura o aro/calota (círculo claro dentro do pneu escuro). O raio do aro
    # fica entre 3% e 6,5% do comprimento do carro e as rodas ficam na metade de baixo.
    min_r, max_r = int(cw * 0.03), int(cw * 0.065)
    rodas = []
    # Diminui o limiar do acumulador até encontrar pelo menos 2 rodas.
    for param2 in range(70, 19, -5):
        circulos = cv2.HoughCircles(suave, cv2.HOUGH_GRADIENT, dp=1, minDist=cw * 0.3,
                                    param1=120, param2=param2,
                                    minRadius=min_r, maxRadius=max_r)
        if circulos is None:
            continue
        rodas = [c for c in np.round(circulos[0]).astype(int)
                 if c[1] > cy + ch * 0.5 and cx <= c[0] <= cx + cw]
        if len(rodas) >= 2:
            break
    return sorted(rodas, key=lambda c: c[0])


def main():
    for caminho in arquivos_entrada(["car.jpg", "car2.png"]):
        processar(caminho)


def processar(caminho):
    print(f"Problema 1 - entrada: {caminho}")
    img = ler_imagem(caminho)
    rodas = detectar_rodas(img)

    saida = img.copy()
    linhas = [f"Arquivo: {os.path.basename(caminho)}", f"Total de rodas encontradas: {len(rodas)}", ""]
    for i, (x, y, r) in enumerate(rodas, 1):
        cv2.circle(saida, (x, y), r, (0, 255, 0), 3)
        cv2.circle(saida, (x, y), 3, (0, 0, 255), -1)
        escrever(saida, f"roda{i}", (x - r, y - r - 6))
        linhas.append(f"Roda {i}: x={x}, y={y}, raio={r}")
    escrever(saida, f"Rodas: {len(rodas)}", (10, 25), escala=0.7, espessura=2)

    salvar_imagem(f"p1_{nome_base(caminho)}_rodas.png", saida)
    salvar_texto(f"p1_{nome_base(caminho)}_rodas.txt", linhas)
    print(f"  {len(rodas)} roda(s) detectada(s)")


if __name__ == "__main__":
    main()
