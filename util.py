"""Funções auxiliares compartilhadas pelos problemas."""
import glob
import os
import sys

import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(BASE, "resultados")


def localizar_arquivo(candidatos, pastas=("", "video", "imagens")):
    """Retorna o primeiro arquivo existente entre os candidatos.

    Se um caminho for passado na linha de comando (sys.argv[1]) ele tem prioridade.
    Os candidatos podem conter curingas (ex.: "*.mp4").
    """
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]
    for nome in candidatos:
        for pasta in pastas:
            for raiz in (BASE, os.getcwd()):
                achados = sorted(glob.glob(os.path.join(raiz, pasta, nome)))
                if achados:
                    return achados[0]
    raise FileNotFoundError(
        f"Nenhum arquivo encontrado entre {candidatos}. "
        "Informe o caminho: python <script>.py caminho/do/arquivo"
    )


def arquivos_entrada(candidatos):
    """Caminho passado na linha de comando, ou todos os candidatos que existirem."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return [sys.argv[1]]
    achados = []
    for nome in candidatos:
        for raiz in (BASE, os.getcwd()):
            caminho = os.path.join(raiz, nome)
            if os.path.exists(caminho):
                achados.append(caminho)
                break
    if not achados:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado entre {candidatos}. "
            "Informe o caminho: python <script>.py caminho/do/arquivo"
        )
    return achados


def nome_base(caminho):
    return os.path.splitext(os.path.basename(caminho))[0]


def ler_imagem(caminho):
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho}")
    return img


def caminho_saida(nome):
    os.makedirs(SAIDA, exist_ok=True)
    return os.path.join(SAIDA, nome)


def salvar_imagem(nome, img):
    caminho = caminho_saida(nome)
    cv2.imwrite(caminho, img)
    print(f"  -> {caminho}")
    return caminho


def salvar_texto(nome, linhas):
    caminho = caminho_saida(nome)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    print(f"  -> {caminho}")
    return caminho


def escrever(img, texto, org, cor=(0, 0, 255), escala=0.45, espessura=1):
    """Escreve texto com contorno branco para ficar legível sobre qualquer fundo."""
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255),
                espessura + 2, cv2.LINE_AA)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, espessura, cv2.LINE_AA)
