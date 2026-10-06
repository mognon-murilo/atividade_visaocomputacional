"""Menu único para executar os 5 problemas de extração de características."""
import sys

import problema1_rodas
import problema2_via
import problema3_celulas
import problema4_folhas
import problema5_ra

OPCOES = {
    "1": ("Problema 1 - Contar rodas (car.jpg)", problema1_rodas.main),
    "2": ("Problema 2 - Linhas da via (via.bmp)", problema2_via.main),
    "3": ("Problema 3 - Elementos do microscópio (analise.jpg)", problema3_celulas.main),
    "4": ("Problema 4 - Folhas verdes x amarelas (folhasdemaca.jpg)", problema4_folhas.main),
    "5": ("Problema 5 - Realidade Aumentada (video/teste.mp4)", problema5_ra.main),
}


def main():
    sys.argv = sys.argv[:1]  # cada problema usa o arquivo padrão
    while True:
        print("\n=== Extração de Características - OpenCV ===")
        for k, (nome, _) in OPCOES.items():
            print(f"  {k}) {nome}")
        print("  t) Executar todos   0) Sair")
        op = input("Opção: ").strip().lower()
        if op == "0":
            break
        escolhidas = list(OPCOES) if op == "t" else [op]
        for k in escolhidas:
            if k not in OPCOES:
                print("Opção inválida.")
                continue
            try:
                OPCOES[k][1]()
            except Exception as erro:  # mostra o erro e volta ao menu
                print(f"  Erro: {erro}")


if __name__ == "__main__":
    main()
