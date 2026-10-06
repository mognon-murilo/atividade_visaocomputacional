# Extração de Características - OpenCV (PDI aula 07)

Soluções dos exercícios (Problemas 1 a 5) da aula de Extração de Características.

## Organização

```
├── video/teste.mp4          # Problema 5
├── car.jpg, car2.png        # Problema 1
├── via.bmp                  # Problema 2
├── analise.jpg              # Problema 3 (bloodcellsdog.jpg também é processada)
├── 147735485.jpg            # Problema 4 (folhas amarela e verde)
├── main.py                  # menu com todos os problemas
├── util.py                  # funções comuns (leitura, gravação, texto)
├── problema1_rodas.py ... problema5_ra.py
└── resultados/              # saídas geradas
```

## Execução

```bash
pip install -r requirements.txt
python main.py                 # menu: 1 a 5, t = todos, 0 = sair
python problema1_rodas.py      # ou cada problema separado
python problema1_rodas.py outra_imagem.jpg   # usando outra imagem de entrada
python problema5_ra.py video/teste.mp4 --overlay piaui.jpg   # projeta outra imagem
python problema5_ra.py --sem-janela          # só grava os vídeos, sem abrir janelas
```

## Soluções e resultados

| Problema | Técnica | Saídas em `resultados/` | Resultado |
|---|---|---|---|
| 1 - Rodas | Recorte do carro + `HoughCircles` (aro dentro do pneu, metade inferior) | `p1_car_rodas.png/.txt`, `p1_car2_rodas.png/.txt` (total, x, y, raio) | 2 rodas em cada carro |
| 2 - Via | Gaussiano + `Canny` com limiares automáticos; extra: tinta branca/amarela junto ao asfalto + `HoughLinesP` | `p2_binarizada.png`, `p2_rotulada.png` (bordas em vermelho), `p2_linhas.png` | bordas e faixas realçadas |
| 3 - Microscópio | Otsu + morfologia + preenchimento de buracos + transformada de distância (sementes) para separar células encostadas | `p3_analise_celulas.png/.txt`, `p3_bloodcellsdog_celulas.png/.txt` (id, área) | 153 e 49 elementos |
| 4 - Folhas | Segmentação por cor em HSV (amarelo e verde; fundo marrom/branco descartado) + classificação pela cor predominante | `p4_folhas.png`, `p4_folhas.txt` (listas amarelas/verdes com id e área) | 1 amarela, 1 verde |
| 5 - RA | `Canny` → contornos → fecho convexo + `approxPolyDP` (quadrilátero) → validação do padrão do marcador → `cornerHarris` (refina os 4 cantos) → `findHomography`/`warpPerspective` | `p5_ra.mp4` (destaque alinhado + 4 cantos), `p5_canny_harris.mp4`; janelas "Canny + Harris" e "Saida RA" (ESC ou q para sair) | marcador rastreado em todo o vídeo, exceto quando a mão o cobre ou o cartão está virado |
