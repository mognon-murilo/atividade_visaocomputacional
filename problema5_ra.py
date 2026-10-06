"""Problema 5 - Realidade Aumentada com marcador plano (video/teste.mp4).

Pipeline por quadro:
  1. Canny  -> bordas do marcador
  2. Contornos + approxPolyDP -> maior quadrilátero convexo (o marcador)
  3. Harris -> refina os 4 vértices para o canto de Harris mais forte próximo
  4. Homografia -> projeta uma imagem/destaque sobre o marcador

Saídas:
  (a) resultados/p5_ra.mp4 - vídeo processado (imagem alinhada + 4 cantos de Harris)
      resultados/p5_canny_harris.mp4 - vídeo com a visualização Canny+Harris
  (b) janelas em tempo de execução: "Canny + Harris" e "Saida RA" (ESC/q encerra)

Uso: python problema5_ra.py [video] [--overlay imagem.jpg] [--sem-janela]
"""
import argparse

import cv2
import numpy as np

from util import caminho_saida, escrever, localizar_arquivo


def criar_overlay(tam=400):
    """Destaque padrão: quadrado azul com texto."""
    img = np.full((tam, tam, 3), (255, 140, 30), np.uint8)
    cv2.rectangle(img, (0, 0), (tam - 1, tam - 1), (255, 255, 255), 8)
    for i, txt in enumerate(["UPF", "Visao", "Computacional"]):
        tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0][0]
        cv2.putText(img, txt, ((tam - tw) // 2, 150 + i * 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def ordenar_cantos(pts):
    """Ordena 4 pontos como: sup-esq, sup-dir, inf-dir, inf-esq."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def encontrar_marcador(bordas, area_min):
    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    melhor, melhor_area = None, area_min
    for c in contornos:
        peri = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(aprox) != 4 or not cv2.isContourConvex(aprox):
            continue
        area = cv2.contourArea(aprox)
        x, y, w, h = cv2.boundingRect(aprox)
        if area > melhor_area and 0.3 < w / float(h) < 3.0:
            melhor, melhor_area = aprox, area
    return melhor


def refinar_com_harris(harris, cantos, raio):
    """Move cada vértice para o ponto de maior resposta de Harris na vizinhança."""
    h, w = harris.shape
    refinados = []
    for x, y in cantos:
        x0, y0 = max(int(x) - raio, 0), max(int(y) - raio, 0)
        x1, y1 = min(int(x) + raio + 1, w), min(int(y) + raio + 1, h)
        janela = harris[y0:y1, x0:x1]
        if janela.size and janela.max() > 0:
            dy, dx = np.unravel_index(np.argmax(janela), janela.shape)
            refinados.append((x0 + dx, y0 + dy))
        else:
            refinados.append((x, y))
    return np.array(refinados, np.float32)


def processar(frame, overlay, estado):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Canny
    bordas = cv2.Canny(suave, 50, 150)
    bordas_fechadas = cv2.dilate(bordas, np.ones((3, 3), np.uint8), iterations=1)

    # 2. Harris (vértices)
    harris = cv2.cornerHarris(np.float32(suave), blockSize=3, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)

    vis = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    vis[harris > 0.01 * harris.max()] = (0, 0, 255)

    area_min = 0.002 * frame.shape[0] * frame.shape[1]
    quad = encontrar_marcador(bordas_fechadas, area_min)
    saida = frame.copy()

    if quad is not None:
        cantos = ordenar_cantos(quad)
        raio = max(4, int(0.02 * max(frame.shape[:2])))
        cantos = refinar_com_harris(harris, cantos, raio)
        # Suaviza o tremor entre quadros.
        if estado.get("cantos") is not None:
            cantos = 0.6 * cantos + 0.4 * estado["cantos"]
        estado["cantos"], estado["perdido"] = cantos, 0
    else:
        estado["perdido"] = estado.get("perdido", 0) + 1
        if estado["perdido"] > 5:
            estado["cantos"] = None

    cantos = estado.get("cantos")
    if cantos is not None:
        # 3. Homografia: cantos do overlay -> cantos do marcador
        oh, ow = overlay.shape[:2]
        origem = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
        H, _ = cv2.findHomography(origem, cantos)
        if H is not None:
            projetada = cv2.warpPerspective(overlay, H, (frame.shape[1], frame.shape[0]))
            mascara = cv2.warpPerspective(np.full((oh, ow), 255, np.uint8), H,
                                          (frame.shape[1], frame.shape[0]))
            saida[mascara > 0] = projetada[mascara > 0]

        for i, (x, y) in enumerate(cantos.astype(int)):
            for img in (saida, vis):
                cv2.circle(img, (int(x), int(y)), 7, (0, 255, 255), 2)
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            escrever(saida, f"C{i + 1}", (int(x) + 8, int(y) - 8), cor=(0, 0, 255))
        cv2.polylines(vis, [cantos.astype(np.int32)], True, (0, 255, 0), 2)
    else:
        escrever(saida, "Marcador nao encontrado", (10, 30), escala=0.8, espessura=2)

    return vis, saida


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?")
    parser.add_argument("--overlay", help="imagem a ser projetada sobre o marcador")
    parser.add_argument("--sem-janela", action="store_true", help="não exibe as janelas")
    args = parser.parse_args()

    caminho = args.video or localizar_arquivo(["teste.mp4", "*.mp4", "*.avi", "*.mov"])
    print(f"Problema 5 - entrada: {caminho}")
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {caminho}")

    overlay = cv2.imread(args.overlay) if args.overlay else criar_overlay()
    if overlay is None:
        raise FileNotFoundError(f"Não foi possível abrir o overlay: {args.overlay}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_ra = cv2.VideoWriter(caminho_saida("p5_ra.mp4"), fourcc, fps, (w, h))
    out_ch = cv2.VideoWriter(caminho_saida("p5_canny_harris.mp4"), fourcc, fps, (w, h))

    estado, quadros, detectados = {}, 0, 0
    atraso = max(1, int(1000 / fps))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis, saida = processar(frame, overlay, estado)
        quadros += 1
        detectados += estado.get("cantos") is not None
        out_ra.write(saida)
        out_ch.write(vis)
        if not args.sem_janela:
            cv2.imshow("Canny + Harris", vis)
            cv2.imshow("Saida RA", saida)
            if cv2.waitKey(atraso) & 0xFF in (27, ord("q")):
                break

    cap.release()
    out_ra.release()
    out_ch.release()
    if not args.sem_janela:
        cv2.destroyAllWindows()
    print(f"  -> {caminho_saida('p5_ra.mp4')}")
    print(f"  -> {caminho_saida('p5_canny_harris.mp4')}")
    print(f"  marcador rastreado em {detectados}/{quadros} quadros")


if __name__ == "__main__":
    main()
