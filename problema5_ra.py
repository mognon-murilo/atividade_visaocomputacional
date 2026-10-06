"""Problema 5 - Realidade Aumentada com marcador plano (video/teste.mp4).

Pipeline por quadro:
  1. Canny  -> bordas; contornos fechados viram candidatos a marcador
  2. Fecho convexo + approxPolyDP -> quadriláteros (tolera um dedo sobre a borda)
  3. Validação: o quadrado preto do marcador tem moldura escura e padrão claro dentro
     (descarta o verso branco do cartão e outros retângulos)
  4. Harris -> refina os 4 vértices para o canto de Harris mais forte próximo
  5. Homografia -> projeta uma imagem/destaque sobre o marcador

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

TAM_VALIDACAO = 64
# O destaque é projetado um pouco maior que o quadrado preto, cobrindo a moldura branca.
MARGEM_OVERLAY = 0.15
# Quantos quadros o último marcador é mantido quando ele some (ex.: mão por cima).
QUADROS_MEMORIA = 8


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


def alinhar_com_anterior(cantos, anteriores):
    """Gira a ordem dos cantos para casar com o quadro anterior (evita o destaque 'girar')."""
    if anteriores is None:
        return cantos
    opcoes = [np.roll(cantos, k, axis=0) for k in range(4)]
    return min(opcoes, key=lambda c: np.linalg.norm(c - anteriores, axis=1).sum())


def eh_marcador(gray, cantos):
    """Retifica o quadrilátero e confere se ele parece o quadrado preto do marcador."""
    t = TAM_VALIDACAO
    destino = np.array([[0, 0], [t - 1, 0], [t - 1, t - 1], [0, t - 1]], np.float32)
    H = cv2.getPerspectiveTransform(cantos, destino)
    quadrado = cv2.warpPerspective(gray, H, (t, t))
    if int(quadrado.max()) - int(quadrado.min()) < 60:
        return False  # sem contraste: verso branco do cartão, mesa, etc.
    limiar = (int(quadrado.min()) + int(quadrado.max())) / 2
    escuro = quadrado < limiar
    m = t // 8
    moldura = np.concatenate([escuro[:m].ravel(), escuro[-m:].ravel(),
                              escuro[m:-m, :m].ravel(), escuro[m:-m, -m:].ravel()])
    miolo = escuro[2 * m:-2 * m, 2 * m:-2 * m]
    # Moldura preta e miolo com padrão (parte clara e parte escura).
    return moldura.mean() > 0.75 and 0.1 < miolo.mean() < 0.9


def encontrar_marcador(bordas, gray, area_min):
    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    melhor, melhor_area = None, area_min
    for c in contornos:
        # O fecho convexo "tapa" entalhes causados por um dedo sobre a borda.
        casco = cv2.convexHull(c)
        peri = cv2.arcLength(casco, True)
        aprox = cv2.approxPolyDP(casco, 0.04 * peri, True)
        if len(aprox) != 4:
            continue
        area = cv2.contourArea(aprox)
        if area <= melhor_area:
            continue
        x, y, w, h = cv2.boundingRect(aprox)
        if not 0.3 < w / float(h) < 3.0:
            continue
        cantos = ordenar_cantos(aprox)
        if eh_marcador(gray, cantos):
            melhor, melhor_area = cantos, area
    return melhor


def refinar_com_harris(harris, cantos, raio):
    """Move cada vértice para o ponto de maior resposta de Harris na vizinhança."""
    h, w = harris.shape
    refinados = []
    for x, y in cantos:
        x0, y0 = max(int(x) - raio, 0), max(int(y) - raio, 0)
        x1, y1 = min(int(x) + raio + 1, w), min(int(y) + raio + 1, h)
        janela = harris[y0:y1, x0:x1]
        if janela.size and janela.max() > 0.01 * harris.max():
            dy, dx = np.unravel_index(np.argmax(janela), janela.shape)
            refinados.append((x0 + dx, y0 + dy))
        else:
            refinados.append((x, y))
    return np.array(refinados, np.float32)


def projetar(saida, overlay, cantos):
    """Homografia: cantos do overlay -> área do marcador (com margem) no quadro."""
    oh, ow = overlay.shape[:2]
    m = MARGEM_OVERLAY
    unitario = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    H_marcador = cv2.getPerspectiveTransform(unitario, cantos.astype(np.float32))
    expandido = np.array([[[-m, -m], [1 + m, -m], [1 + m, 1 + m], [-m, 1 + m]]], np.float32)
    destino = cv2.perspectiveTransform(expandido, H_marcador)[0]

    origem = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
    H, _ = cv2.findHomography(origem, destino)
    if H is None:
        return
    tam = (saida.shape[1], saida.shape[0])
    projetada = cv2.warpPerspective(overlay, H, tam)
    mascara = cv2.warpPerspective(np.full((oh, ow), 255, np.uint8), H, tam)
    saida[mascara > 0] = projetada[mascara > 0]


def processar(frame, overlay, estado):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Canny (bordas)
    bordas = cv2.Canny(suave, 50, 150)
    bordas_fechadas = cv2.dilate(bordas, np.ones((3, 3), np.uint8), iterations=1)

    # 2. Harris (vértices)
    harris = cv2.cornerHarris(np.float32(suave), blockSize=3, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)

    vis = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    vis[harris > 0.01 * harris.max()] = (0, 0, 255)

    area_min = 0.002 * frame.shape[0] * frame.shape[1]
    cantos = encontrar_marcador(bordas_fechadas, suave, area_min)
    anteriores = estado.get("cantos")

    if cantos is not None:
        raio = max(4, int(0.015 * max(frame.shape[:2])))
        cantos = refinar_com_harris(harris, cantos, raio)
        cantos = alinhar_com_anterior(cantos, anteriores)
        # Suaviza o tremor entre quadros (só se o marcador não "pulou").
        if anteriores is not None and np.abs(cantos - anteriores).max() < 40:
            cantos = 0.7 * cantos + 0.3 * anteriores
        estado["cantos"], estado["perdido"] = cantos, 0
        estado["detectados"] = estado.get("detectados", 0) + 1
    else:
        estado["perdido"] = estado.get("perdido", 0) + 1
        if estado["perdido"] > QUADROS_MEMORIA:
            estado["cantos"] = None

    saida = frame.copy()
    cantos = estado.get("cantos")
    if cantos is not None:
        # 3. Homografia (projeção do destaque)
        projetar(saida, overlay, cantos)
        for i, (x, y) in enumerate(cantos.astype(int)):
            for img in (saida, vis):
                cv2.circle(img, (int(x), int(y)), 7, (0, 255, 255), 2)
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            escrever(saida, f"C{i + 1}", (int(x) + 8, int(y) - 8), cor=(0, 0, 255))
        cv2.polylines(vis, [cantos.astype(np.int32)], True, (0, 255, 0), 2)
    else:
        escrever(saida, "Marcador nao encontrado", (10, 30), escala=0.8, espessura=2)

    escrever(vis, "Canny + Harris", (10, 25), cor=(0, 255, 0), escala=0.7, espessura=2)
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

    estado, quadros = {}, 0
    atraso = max(1, int(1000 / fps))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis, saida = processar(frame, overlay, estado)
        quadros += 1
        out_ra.write(saida)
        out_ch.write(vis)
        if not args.sem_janela:
            try:
                cv2.imshow("Canny + Harris", vis)
                cv2.imshow("Saida RA", saida)
            except cv2.error:
                print("  Aviso: OpenCV sem suporte a janelas; gravando apenas os vídeos.")
                args.sem_janela = True
                continue
            if cv2.waitKey(atraso) & 0xFF in (27, ord("q")):
                break

    cap.release()
    out_ra.release()
    out_ch.release()
    if not args.sem_janela:
        cv2.destroyAllWindows()
    print(f"  -> {caminho_saida('p5_ra.mp4')}")
    print(f"  -> {caminho_saida('p5_canny_harris.mp4')}")
    print(f"  marcador detectado em {estado.get('detectados', 0)}/{quadros} quadros")


if __name__ == "__main__":
    main()
