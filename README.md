# MS671 - Reconhecimento de Objetos Usando YOLO

Este repositório contém o código-fonte, os experimentos e o relatório final do projeto da disciplina **MS671 - Introdução ao Aprendizado Profundo** (UNICAMP). O trabalho consiste na implementação, análise arquitetural e *benchmarking* computacional da rede de detecção de objetos **YOLOv3**, com comparações empíricas contra implementações nativas em C++ (Torchvision) e a rede YOLOv11 Nano.

**Autores:** Victor Luigi Roquetto & Isabel Cristina Marras Salles

---

## 📄 Relatório Técnico
Toda a fundamentação teórica, a matemática geométrica (*Anchor Boxes* e *IoU*) e as discussões estatísticas dos resultados estão documentadas no relatório oficial do projeto.

👉 **[Ler o Relatório Completo (PDF)](Ms671_Porjeto1.pdf)**

---

## 🧠 Arquitetura da Rede (YOLOv3)

A implementação do modelo neste projeto foi dividida estruturalmente em três módulos principais que ditam o fluxo do tensor:

* **Backbone (Darknet-53):** Atua como o motor de extração de características. Utiliza convoluções com *stride* de 2 (no lugar de *max pooling*) e conexões residuais para mapear a imagem original em um espaço latente de alta densidade semântica sem perder a integridade espacial.
* **Neck (Feature Pyramid Network - FPN):** Responsável por fundir mapas de diferentes resoluções. Através de operações de *upsampling* e concatenações laterais com camadas prévias do Backbone, permite que o modelo recupere minúcias visuais, garantindo a detecção robusta de objetos de variados tamanhos.
* **Head (YOLO Layers):** Realiza as predições finais de forma paralela em três escalas espaciais (13x13, 26x26, 52x52). Converte os tensores processados em coordenadas de *Bounding Boxes*, níveis de confiança da presença de objetos e probabilidades multi-classes (utilizando *Sigmoid* no lugar de *Softmax*).
* **Pós-processamento (Non-Maximum Suppression - NMS):** Algoritmo (implementado manualmente no projeto) que calcula a Intersecção sobre a União (*IoU*) para filtrar e suprimir *Anchor Boxes* redundantes, mantendo apenas a predição mais acurada para cada objeto.

---

## 🔬 Experimentos e Testes Realizados

A análise experimental atestou a robustez da arquitetura em múltiplos cenários e promoveu um rigoroso *micro-benchmarking* computacional:

1. **Ajuste de Limiares (*Thresholds*):** Análise do comportamento da rede mediante variações de níveis de confiança e cortes de IoU, evidenciando o compromisso entre a supressão de poluição visual e a omissão de objetos em cenários de alta oclusão.
2. **Robustez a Degradações (Imagens Modificadas):** Testes de estresse submetendo a rede a imagens com alto ruído, sub/superexposição de luz, desfoque de movimento (*motion blur*) e monocromatismo.
3. **Estabilidade Computacional (NMS Manual vs. Nativo):** Execução de *benchmarks* intra-imagem isolando a função NMS. Demonstrou-se a variância provocada pelo gerenciamento de memória (*Garbage Collector*) e natureza sequencial do Python versus a estabilidade absoluta da implementação paralela nativa em C++/CUDA (*Torchvision*).
4. **Pipeline Inter-imagens (YOLOv3 vs. YOLO11n):** Comparação do tempo real de inferência global em um lote heterogêneo de imagens. O modelo YOLOv11 Nano atestou a evolução arquitetural recente, executando as predições em uma fração do tempo exigido pelo modelo clássico.

---

## 📂 Estrutura de Diretórios

O repositório está organizado da seguinte forma:

```text
├── Ms671_Porjeto1.pdf              # Relatório técnico final documentando toda a pesquisa
├── trabalho1/
│   ├── teste_eficiencia.py         # Script principal de inferência, predição e benchmark (YOLOv3 e v11)
│   ├── utils.py                    # Algoritmo de NMS e IoU desenvolvidos manualmente
│   ├── YOLO_Pytorch.ipynb          # Notebook de prototipagem e desenvolvimento da arquitetura
│   ├── cfg/                        # Configurações originais da rede Darknet (yolov3.cfg)
│   ├── data/                       # Nomenclaturas das classes do dataset COCO
│   ├── weights/                    # Pesos (weights) originais em C e binários convertidos para PyTorch (.pth)
│   │
│   ├── images/                     # Dataset original de validação
│   ├── images_variations/          # Dataset modificado (ruído, desfoque, exposição) para testes de robustez
│   │
│   ├── results_general/            # Imagens de saída com detecções do pipeline padrão
│   ├── results_iou/                # Imagens de saída evidenciando variações do threshold de IoU
│   └── results_modificadas/        # Imagens de saída testando detecções nas imagens degradadas
│
├── graficos/
│   ├── gerador.py                  # Script para plotagem dos gráficos estáticos do relatório
│   └── *.png                       # Gráficos renderizados (Evolução YOLO, Tempos de NMS, etc.)
│
├── gerar_latex.py                  # Script auxiliar para automatização de tabelas e outputs
└── requirements.txt                # Dependências do projeto (PyTorch, Ultralytics, etc.)
