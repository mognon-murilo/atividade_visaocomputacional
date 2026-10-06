"""Problema 3 - Analisa: contar os elementos de uma amostra de microscópio (analise.jpg).

Saídas:
  (a) resultados/p3_celulas.png - imagem rotulada (retângulo + id em cada elemento)
  (b) resultados/p3_celulas.txt - total e lista enumerada (id, área em pixels)
"""
import cv2
import numpy as np

from util import escrever, ler_imagem, localizar_arquivo, salvar_imagem, salvar_texto


def preencher_buracos(mask):
    """Preenche o centro claro das hemácias (buracos dentro dos objetos)."""
    # Moldura de 1 px garante que a inundação comece no fundo.
    inundada = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(inundada, None, (0, 0), 255)
    return mask | cv2.bitwise_not(inundada[1:-1, 1:-1])


def segmentar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Células são mais escuras que o fundo -> limiar de Otsu invertido.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = preencher_buracos(mask)

    # Separa células que se tocam: transformada de distância + sementes.
    # Cada célula vira uma semente no máximo local da transformada de distância.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    n_comp, comp = cv2.connectedComponents(mask)
    maximos = [dist[comp == i].max() for i in range(1, n_comp)]
    raio = max(3.0, float(np.median(maximos))) if maximos else 3.0
    tam = int(raio * 1.2) | 1
    vizinhanca = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam, tam)))
    picos = ((dist >= vizinhanca) & (dist > 0.5 * raio)).astype(np.uint8) * 255
    # Junta picos muito próximos (platôs) em uma única semente.
    sementes = cv2.dilate(picos, kernel, iterations=2)

    # Cada pixel do objeto pertence à semente mais próxima (partição de Voronoi
    # restrita à máscara), o que separa células encostadas umas nas outras.
    _, rotulos = cv2.distanceTransformWithLabels(255 - sementes, cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_CCOMP)
    rotulos[mask == 0] = 0

    elementos = []
    for rotulo in np.unique(rotulos):
        if rotulo == 0:
            continue
        regiao = (rotulos == rotulo).astype(np.uint8)
        # Mantém apenas o maior pedaço conexo da região.
        n, comp, stats, _ = cv2.connectedComponentsWithStats(regiao)
        if n < 2:
            continue
        maior = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[maior]
        elementos.append({"area": int(area), "rect": (int(x), int(y), int(w), int(h))})

    # Remove ruídos muito pequenos em relação ao tamanho típico de uma célula.
    if elementos:
        mediana = np.median([e["area"] for e in elementos])
        elementos = [e for e in elementos if e["area"] >= 0.2 * mediana]
    # Ordena de cima para baixo, esquerda para direita.
    elementos.sort(key=lambda e: (e["rect"][1] // 20, e["rect"][0]))
    return elementos


def main():
    caminho = localizar_arquivo(["analise.jpg", "bloodcellsdog.jpg"])
    print(f"Problema 3 - entrada: {caminho}")
    img = ler_imagem(caminho)
    elementos = segmentar(img)

    saida = img.copy()
    linhas = [f"Arquivo: {caminho}", f"Total de elementos encontrados: {len(elementos)}", "",
              "id;area_pixels"]
    for i, e in enumerate(elementos, 1):
        x, y, w, h = e["rect"]
        cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 0, 255), 1)
        escrever(saida, str(i), (x + 2, y + 12), escala=0.35)
        linhas.append(f"{i};{e['area']}")

    salvar_imagem("p3_celulas.png", saida)
    salvar_texto("p3_celulas.txt", linhas)
    print(f"  {len(elementos)} elemento(s) detectado(s)")


if __name__ == "__main__":
    main()
