# Extração de Características - OpenCV (PDI aula 07)

Soluções dos exercícios (Problemas 1 a 5) da aula de Extração de Características.

## Organização esperada

Copie os scripts `.py` para a mesma pasta das imagens:

```
ATIVIDADE_/
├── video/teste.mp4          # Problema 5
├── analise.jpg              # Problema 3
├── car.jpg                  # Problema 1
├── via.bmp                  # Problema 2
├── folhasdemaca.jpg         # Problema 4 (ou istockphoto-modified.jpg)
├── main.py, util.py, problema1_rodas.py ... problema5_ra.py
```

## Execução

```bash
pip install -r requirements.txt
python main.py                 # menu com todos os problemas
python problema1_rodas.py      # ou cada problema separado
python problema1_rodas.py car2.png   # usando outra imagem de entrada
python problema5_ra.py video/teste.mp4 --overlay piaui.jpg
```

Todas as saídas são gravadas em `resultados/`.

| Problema | Técnica | Saídas |
|---|---|---|
| 1 - Rodas | `HoughCircles` (círculos na metade inferior) | `p1_rodas.png`, `p1_rodas.txt` (total, x, y, raio) |
| 2 - Via | Gaussiano + `Canny` (+ `HoughLinesP` extra) | `p2_binarizada.png`, `p2_rotulada.png` (bordas em vermelho), `p2_linhas.png` |
| 3 - Microscópio | Otsu + morfologia + transformada de distância (separa células encostadas) | `p3_celulas.png`, `p3_celulas.txt` (id, área) |
| 4 - Folhas | Segmentação por saturação (HSV) + classificação pela matiz | `p4_folhas.png`, `p4_folhas.txt` (listas amarelas/verdes com id e área) |
| 5 - RA | `Canny` + contorno quadrilátero + `cornerHarris` + `findHomography`/`warpPerspective` | `p5_ra.mp4`, `p5_canny_harris.mp4` e janelas "Canny + Harris" / "Saida RA" (ESC ou q para sair) |
