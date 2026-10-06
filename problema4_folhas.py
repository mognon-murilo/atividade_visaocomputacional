"""Problema 4 - Classificar folhas em "tons de verde" e "tons de amarelo" (folhasdemaca.jpg).

Saídas:
  (a) resultados/p4_folhas.png - imagem rotulada (retângulo + verdeN / amarelaN)
  (b) resultados/p4_folhas.txt - totais e as listas enumeradas (id, área em pixels)
"""
import cv2
import numpy as np

from util import escrever, ler_imagem, localizar_arquivo, salvar_imagem, salvar_texto

# Faixas de matiz (H no OpenCV vai de 0 a 179).
AMARELO = (8, 33)    # amarelo/laranja/marrom
VERDE = (34, 95)


def segmentar_folhas(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Fundo claro e pouco saturado; folhas são saturadas (ou escuras).
    mask = ((hsv[..., 1] > 50) | (hsv[..., 2] < 90)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = 0.002 * img.shape[0] * img.shape[1]
    return hsv, [c for c in contornos if cv2.contourArea(c) >= area_min]


def classificar(hsv, contorno):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(mask, [contorno], -1, 255, -1)
    h = hsv[..., 0][(mask > 0) & (hsv[..., 1] > 40)]
    amarelos = np.count_nonzero((h >= AMARELO[0]) & (h <= AMARELO[1]))
    verdes = np.count_nonzero((h >= VERDE[0]) & (h <= VERDE[1]))
    return ("amarela" if amarelos > verdes else "verde"), int(np.count_nonzero(mask))


def main():
    caminho = localizar_arquivo(["folhasdemaca.jpg", "istockphoto-modified.jpg", "147735485.jpg"])
    print(f"Problema 4 - entrada: {caminho}")
    img = ler_imagem(caminho)
    hsv, contornos = segmentar_folhas(img)

    # Ordena de cima para baixo, esquerda para direita.
    contornos = sorted(contornos, key=lambda c: (cv2.boundingRect(c)[1] // 40,
                                                 cv2.boundingRect(c)[0]))
    listas = {"verde": [], "amarela": []}
    saida = img.copy()
    for c in contornos:
        classe, area = classificar(hsv, c)
        listas[classe].append(area)
        nome = f"{classe}{len(listas[classe])}"
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 0, 255), 2)
        escrever(saida, nome, (x + 3, y + 15), escala=0.5)

    linhas = [f"Arquivo: {caminho}",
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
