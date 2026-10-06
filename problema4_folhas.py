"""Problema 4 - Classificar folhas em "tons de verde" e "tons de amarelo".

Entrada padrão: folhasdemaca.jpg (ou 147735485.jpg / istockphoto-modified.jpg).

Saídas:
  (a) resultados/p4_folhas.png - imagem rotulada (retângulo + verdeN / amarelaN)
  (b) resultados/p4_folhas.txt - totais e as listas enumeradas (id, área em pixels)
"""
import os

import cv2
import numpy as np

from util import escrever, ler_imagem, localizar_arquivo, salvar_imagem, salvar_texto

# Faixas de cor em HSV (no OpenCV H vai de 0 a 179).
# Marrons/laranjas (H < 18) e brancos (S baixa) ficam de fora -> fundo.
AMARELO = ((18, 80, 110), (34, 255, 255))
VERDE = ((35, 45, 30), (90, 255, 255))


def preencher_buracos(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cheia = np.zeros_like(mask)
    cv2.drawContours(cheia, contornos, -1, 255, -1)
    return cheia


def segmentar_folhas(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    amarelo = cv2.inRange(hsv, *AMARELO)
    verde = cv2.inRange(hsv, *VERDE)
    mask = amarelo | verde

    # Abertura remove gravetos/agulhas finas; fechamento une nervuras e manchas.
    k = max(5, min(img.shape[:2]) // 100) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = preencher_buracos(mask)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = 0.004 * img.shape[0] * img.shape[1]
    contornos = [c for c in contornos if cv2.contourArea(c) >= area_min]
    return amarelo, verde, contornos


def classificar(amarelo, verde, contorno):
    """Classe pela cor predominante dentro do contorno da folha."""
    mask = np.zeros(amarelo.shape, np.uint8)
    cv2.drawContours(mask, [contorno], -1, 255, -1)
    n_amarelo = cv2.countNonZero(amarelo & mask)
    n_verde = cv2.countNonZero(verde & mask)
    return ("amarela" if n_amarelo > n_verde else "verde"), cv2.countNonZero(mask)


def main():
    caminho = localizar_arquivo(["folhasdemaca.jpg", "147735485.jpg", "istockphoto-modified.jpg"])
    print(f"Problema 4 - entrada: {caminho}")
    img = ler_imagem(caminho)
    amarelo, verde, contornos = segmentar_folhas(img)

    # Ordena de cima para baixo, esquerda para direita.
    contornos = sorted(contornos, key=lambda c: (cv2.boundingRect(c)[1] // 40,
                                                 cv2.boundingRect(c)[0]))
    listas = {"verde": [], "amarela": []}
    saida = img.copy()
    for c in contornos:
        classe, area = classificar(amarelo, verde, c)
        listas[classe].append(area)
        nome = f"{classe}{len(listas[classe])}"
        x, y, w, h = cv2.boundingRect(c)
        esp = max(2, img.shape[1] // 400)
        cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 0, 255), esp)
        escrever(saida, nome, (x + 5, y + 10 + 20 * esp), escala=0.35 * esp, espessura=esp)

    linhas = [f"Arquivo: {os.path.basename(caminho)}",
              f"Total de folhas: {len(contornos)}",
              f"Folhas amarelas: {len(listas['amarela'])}",
              f"Folhas verdes: {len(listas['verde'])}", "",
              "Folhas amarelas (id;area_pixels)"]
    linhas += [f"amarela{i};{a}" for i, a in enumerate(listas["amarela"], 1)]
    linhas += ["", "Folhas verdes (id;area_pixels)"]
    linhas += [f"verde{i};{a}" for i, a in enumerate(listas["verde"], 1)]

    salvar_imagem("p4_folhas.png", saida)
    salvar_texto("p4_folhas.txt", linhas)
    print(f"  {len(listas['amarela'])} amarela(s), {len(listas['verde'])} verde(s)")


if __name__ == "__main__":
    main()
