import os
import random
import time
import colorsys
import numpy as np
import torch
#print("TORCH IMPORTADO")
from ultralytics import YOLO
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from utils import manual_nms  # função manual de NMS e IoU que devem implementar

# ==========================================
# 1. ÂNCORAS DO YOLOV3 (COCO)
# ==========================================
YOLOV3_ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],  # Escala 1 (13x13) - Objetos Maiores
    [(30, 61), (62, 45), (59, 119)],      # Escala 2 (26x26) - Objetos Médios
    [(10, 13), (16, 30), (33, 23)]        # Escala 3 (52x52) - Objetos Menores
]

# ==========================================
# 2. FUNÇÕES DE UTILIDADE E PRÉ-PROCESSAMENTO
# ==========================================
def read_classes(classes_path):
    with open(classes_path) as f:
        return [c.strip() for c in f.readlines()]

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)
    random.shuffle(colors)
    random.seed(None)
    return colors

def letterbox_image(image, size=(416, 416)):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

def reverter_escala_caixas(boxes, img_size, original_shape):
    iw, ih = original_shape
    w, h = img_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    dx = (w - nw) / 2.0 / w
    dy = (h - nh) / 2.0 / h
    scale_w = nw / w
    scale_h = nh / h
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dy) / scale_h
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dx) / scale_w
    boxes[:, [0, 2]] *= ih
    boxes[:, [1, 3]] *= iw
    return boxes

def preprocess_image(img_path, model_image_size=(416, 416)):
    image = Image.open(img_path).convert('RGB')
    boxed_image = letterbox_image(image, model_image_size)
    image_data = np.array(boxed_image, dtype='float32') / 255.0
    image_data = image_data[:, :, ::-1].copy()
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = torch.from_numpy(image_data).unsqueeze(0)
    return image, image_data

# ==========================================
# 3. BLOCOS DE CONSTRUÇÃO DA REDE
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-5) if bn else None
        self.act = nn.LeakyReLU(0.1, inplace=True) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.act(self.bn(x))
        return x

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, 1, 1, 0)
        self.conv2 = ConvBlock(channels // 2, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = ConvBlock(3, 32, 3, 1, 1)
        self.layer1 = self._make_layer(32, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 8)
        self.layer4 = self._make_layer(256, 512, 8)
        self.layer5 = self._make_layer(512, 1024, 4)

        self.head1_1 = self._make_c5(1024, 512)
        self.head1_2 = self._make_yolo_head(512, 1024, num_classes)

        self.head2_1 = ConvBlock(512, 256, 1, 1, 0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.head2_2 = self._make_c5(768, 256)
        self.head2_3 = self._make_yolo_head(256, 512, num_classes)

        self.head3_1 = ConvBlock(256, 128, 1, 1, 0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.head3_2 = self._make_c5(384, 128)
        self.head3_3 = self._make_yolo_head(128, 256, num_classes)

    def _make_layer(self, in_c, out_c, num_blocks):
        layers = [ConvBlock(in_c, out_c, 3, 2, 1)]
        for _ in range(num_blocks):
            layers.append(ResBlock(out_c))
        return nn.Sequential(*layers)

    def _make_c5(self, in_c, out_c):
        return nn.Sequential(
            ConvBlock(in_c, out_c, 1, 1, 0),
            ConvBlock(out_c, out_c * 2, 3, 1, 1),
            ConvBlock(out_c * 2, out_c, 1, 1, 0),
            ConvBlock(out_c, out_c * 2, 3, 1, 1),
            ConvBlock(out_c * 2, out_c, 1, 1, 0),
        )

    def _make_yolo_head(self, in_c, out_c, num_classes):
        return nn.Sequential(
            ConvBlock(in_c, out_c, 3, 1, 1),
            nn.Conv2d(out_c, 3 * (5 + num_classes), 1, 1, 0, bias=True)
        )

    def forward(self, x):
        x = self.layer2(self.layer1(self.conv1(x)))
        route1 = self.layer3(x)
        route2 = self.layer4(route1)
        x = self.layer5(route2)

        x1_5 = self.head1_1(x)
        out1 = self.head1_2(x1_5)

        x = self.upsample1(self.head2_1(x1_5))
        x = torch.cat([x, route2], dim=1)
        x2_5 = self.head2_2(x)
        out2 = self.head2_3(x2_5)

        x = self.upsample2(self.head3_1(x2_5))
        x = torch.cat([x, route1], dim=1)
        x3_5 = self.head3_2(x)
        out3 = self.head3_3(x3_5)

        return out1, out2, out3

# ==========================================
# 4. CARREGAR PESOS
# ==========================================
def carregar_pesos_yolov3(caminho_weights, modelo):
    with open(caminho_weights, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

    modulos = []
    for m in modelo.modules():
        if isinstance(m, ConvBlock) or (isinstance(m, nn.Conv2d) and m.bias is not None):
            modulos.append(m)

    ptr = 0
    for modulo in modulos:
        if isinstance(modulo, ConvBlock):
            conv, bn = modulo.conv, modulo.bn
            num_bn = bn.bias.numel()
            bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.bias)); ptr += num_bn
            bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.weight)); ptr += num_bn
            bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.running_mean)); ptr += num_bn
            bn_var = torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.running_var)
            bn.running_var.data.copy_(torch.clamp(bn_var, min=1e-5)); ptr += num_bn
            num_w = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_w]).view_as(conv.weight)); ptr += num_w
        elif isinstance(modulo, nn.Conv2d):
            num_b = modulo.bias.numel()
            modulo.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b]).view_as(modulo.bias)); ptr += num_b
            num_w = modulo.weight.numel()
            modulo.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_w]).view_as(modulo.weight)); ptr += num_w
    return modelo

# ==========================================
# 5. YOLO HEAD & PREDIÇÃO
# ==========================================
def decode_yolo(feats, anchors, num_classes, img_size=416):
    B, C, H, W = feats.shape
    num_anchors = len(anchors)
    feats = feats.view(B, num_anchors, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).float().to(feats.device).view(1, 1, H, W, 2)
    box_xy = torch.sigmoid(feats[..., :2])
    box_xy = (box_xy + grid) / torch.tensor([W, H], dtype=torch.float32, device=feats.device)
    anchors_tensor = torch.tensor(anchors, dtype=torch.float32, device=feats.device).view(1, num_anchors, 1, 1, 2)
    box_wh = torch.exp(torch.clamp(feats[..., 2:4], max=15.0)) * anchors_tensor
    box_wh = box_wh / img_size
    box_confidence = torch.sigmoid(feats[..., 4:5])
    box_class_probs = torch.sigmoid(feats[..., 5:])
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    boxes = torch.cat([box_mins[..., 1:2], box_mins[..., 0:1], box_maxes[..., 1:2], box_maxes[..., 0:1]], dim=-1)
    scores = box_confidence * box_class_probs
    return boxes.view(-1, 4), scores.view(-1, num_classes)


def _desenhar_deteccoes(image, boxes, scores, classes, class_names):
    colors = generate_colors(class_names)
    font = ImageFont.load_default()
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(classes.cpu().numpy()))):
        predicted_class = class_names[c]
        box = boxes[i].cpu().numpy()
        score = scores[i].cpu().item()
        label = f'{predicted_class} {score:.2f}'
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), label, font=font)
        label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        top, left, bottom, right = box
        top    = max(0, np.floor(top + 0.5).astype('int32'))
        left   = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right  = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        if left >= right or top >= bottom:
            continue
        text_origin = np.array([left, top - label_size[1]]) if top - label_size[1] >= 0 else np.array([left, top + 1])
        for j in range(thickness):
            if left + j >= right - j or top + j >= bottom - j:
                break
            draw.rectangle([left+j, top+j, right-j, bottom-j], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
        del draw

    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def executar_predicao(image_file, model, class_names, device,
                      score_threshold=0.5, iou_threshold=0.4,
                      show_plot=False, benchmark_nms=False, num_rodadas_nms=1):
    image, image_data = preprocess_image(image_file, (416, 416))
    image_data = image_data.to(device)

    model.eval()
    with torch.no_grad():
        out1, out2, out3 = model(image_data)

        b1, s1 = decode_yolo(out1, YOLOV3_ANCHORS[0], len(class_names), 416)
        b2, s2 = decode_yolo(out2, YOLOV3_ANCHORS[1], len(class_names), 416)
        b3, s3 = decode_yolo(out3, YOLOV3_ANCHORS[2], len(class_names), 416)

        all_boxes  = torch.cat([b1, b2, b3], dim=0)
        all_scores = torch.cat([s1, s2, s3], dim=0)

        valid_mask = torch.isfinite(all_boxes).all(dim=-1) & torch.isfinite(all_scores).all(dim=-1)
        all_boxes  = all_boxes[valid_mask]
        all_scores = all_scores[valid_mask]

        box_class_scores, box_classes = torch.max(all_scores, dim=-1)
        mask = box_class_scores >= score_threshold

        boxes   = all_boxes[mask]
        scores  = box_class_scores[mask]
        classes = box_classes[mask]

        if boxes.size(0) == 0:
            return (0.0, 0.0) if benchmark_nms else None

        boxes = reverter_escala_caixas(boxes, (416, 416), image.size)

        if benchmark_nms:
            tempos_nms_manual = []
            tempos_nms_torch  = []

            for iteracao in range(num_rodadas_nms):
                if num_rodadas_nms > 1 and iteracao % 20 == 0:
                    print(f"  -> Progresso NMS: {iteracao}/{num_rodadas_nms}...", end='\r')

                # Usando time.time() no lugar de perf_counter e sem sincronização CUDA
                start = time.time()
                keep_torch = torchvision.ops.nms(boxes, scores, iou_threshold)
                torchvision_time = (time.time() - start) * 1000
                tempos_nms_torch.append(torchvision_time)

                start = time.time()
                keep_manual = manual_nms(boxes, scores, classes, iou_threshold)
                manual_time = (time.time() - start) * 1000
                tempos_nms_manual.append(manual_time)

                keep = keep_manual[:10]
                boxes_f, scores_f, classes_f = boxes[keep], scores[keep], classes[keep]
            
            if num_rodadas_nms > 1: print(" " * 50, end='\r')

            if show_plot:
                _desenhar_deteccoes(image, boxes_f, scores_f, classes_f, class_names)

            return tempos_nms_manual, tempos_nms_torch

        else:
            keep_manual = manual_nms(boxes, scores, classes, iou_threshold)
            keep = keep_manual[:10]
            boxes_f, scores_f, classes_f = boxes[keep], scores[keep], classes[keep]

            if show_plot:
                _desenhar_deteccoes(image, boxes_f, scores_f, classes_f, class_names)

            return None


# ==========================================
# 6. EXECUÇÃO PRINCIPAL E BENCHMARK EM LOTE
# ==========================================
if __name__ == '__main__':
    device = torch.device("cpu")
    print(f"Processamento via: {device}")

    PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
    class_names = read_classes(os.path.join(PASTA_ATUAL, "data", "coco.names"))

    modelo_v3 = YOLOv3(num_classes=len(class_names)).to(device)
    caminho_pesos_conv = os.path.join(PASTA_ATUAL, "weights", "yolov3_convertido.pth")
    modelo_v3.load_state_dict(torch.load(caminho_pesos_conv, map_location=device))

    print("\nInicializando YOLOv11 (Ultralytics)...")
    modelo_v11 = YOLO('yolo11n.pt')
    

    lista_imagens = [
        "dog.jpg", "eagle.jpg", "giraffe.jpg", "horses.jpg", "person.jpg",
        "city_scene.jpg", "food.jpg", "surf.jpg", "wine.jpg", "motorbike.jpg"
    ]

    nome_img_teste = "city_scene.jpg"
    caminho_img_teste = os.path.join(PASTA_ATUAL, "images", nome_img_teste)

    if not os.path.exists(caminho_img_teste):
        print(f"Erro: Imagem de teste '{nome_img_teste}' não encontrada.")
        exit()

    NUM_RODADAS_INTRA = 200
    WARMUP = 3

    print("\n" + "="*55)
    print(f" BENCHMARK 1 — ESTABILIDADE NMS (intra-imagem)")
    print(f" Imagem : {nome_img_teste}")
    print(f" Rodadas: {NUM_RODADAS_INTRA} total  |  {WARMUP} de aquecimento descartadas")
    print("="*55)

    tempos_nms_manual, tempos_nms_torch = executar_predicao(
        caminho_img_teste, modelo_v3, class_names, device,
        score_threshold=0.5, benchmark_nms=True, num_rodadas_nms=NUM_RODADAS_INTRA
    )
    
    # Descartando as rodadas de aquecimento (warmup)
    tempos_nms_manual = tempos_nms_manual[WARMUP:]
    tempos_nms_torch = tempos_nms_torch[WARMUP:]

    print("Concluído!\n")

    media_man = np.mean(tempos_nms_manual)
    std_man   = np.std(tempos_nms_manual)
    var_man   = np.var(tempos_nms_manual)

    media_torch = np.mean(tempos_nms_torch)
    std_torch   = np.std(tempos_nms_torch)
    var_torch   = np.var(tempos_nms_torch)

    print("="*55)
    print(" ANÁLISE DE ESTABILIDADE COMPUTACIONAL (NMS)")
    print("="*55)
    print("[ NMS MANUAL (Python) ]")
    print(f"  Média    : {media_man:.4f} ms")
    print(f"  Variância: {var_man:.4f} ms²")
    print(f"  Desvio   : ±{std_man:.4f} ms")
    print("\n[ NMS NATIVO (Torchvision C++) ]")
    print(f"  Média    : {media_torch:.4f} ms")
    print(f"  Variância: {var_torch:.4f} ms²")
    print(f"  Desvio   : ±{std_torch:.4f} ms")
    print("="*55)


    tempos_total_v3  = []
    tempos_total_v11 = []
    tempos_nms_manual_inter = []
    tempos_nms_torch_inter  = []

    print("\n" + "="*55)
    print(" BENCHMARK 2 — PIPELINE COMPLETO (inter-imagens)")
    print("="*55)

    for img_name in lista_imagens:
        caminho_img = os.path.join(PASTA_ATUAL, "images", img_name)

        if not os.path.exists(caminho_img):
            print(f"-> Aviso: {img_name} não encontrada. Pulando...")
            continue

        print(f"Processando: {img_name}")

        start_v3 = time.time()
        executar_predicao(
            caminho_img, modelo_v3, class_names, device,
            score_threshold=0.5, benchmark_nms=False 
        )
        total_v3 = (time.time() - start_v3) * 1000
        tempos_total_v3.append(total_v3)

        t_man, t_torch = executar_predicao(
            caminho_img, modelo_v3, class_names, device,
            score_threshold=0.5, benchmark_nms=True, num_rodadas_nms=1
        )
        tempos_nms_manual_inter.append(t_man[0])
        tempos_nms_torch_inter.append(t_torch[0])

        start_v11 = time.time()
        resultados = modelo_v11(caminho_img, verbose=False)
        total_v11 = (time.time() - start_v11) * 1000
        tempos_total_v11.append(total_v11)

    print("\n" + "="*55)
    print(" RELATÓRIO FINAL DE DESEMPENHO (MÉDIAS INTER-IMAGENS)")
    print("="*55)
    print(f"Imagens válidas processadas: {len(tempos_total_v3)}\n")

    print("COMPARAÇÃO 1: Funções de NMS (medição isolada, inter-imagens)")
    print(f"  - NMS Manual (Python)    : {np.mean(tempos_nms_manual_inter):.4f} ms  ±{np.std(tempos_nms_manual_inter):.4f}")
    print(f"  - NMS Nativo (C++/CUDA)  : {np.mean(tempos_nms_torch_inter):.4f} ms  ±{np.std(tempos_nms_torch_inter):.4f}")

    print("\nCOMPARAÇÃO 2: Pipeline Completo (pré-proc + rede + 1x NMS)")
    print(f"  - YOLOv3 Clássico (Manual): {np.mean(tempos_total_v3):.4f} ms  ±{np.std(tempos_total_v3):.4f}")
    print(f"  - YOLOv11 Nano (Ultralytics): {np.mean(tempos_total_v11):.4f} ms  ±{np.std(tempos_total_v11):.4f}")
    print("="*55)

    print("\nGerando gráficos de desempenho...")

    std_nms_manual_intra = np.std(tempos_nms_manual) 
    std_nms_torch_intra  = np.std(tempos_nms_torch)
    std_v3  = np.std(tempos_total_v3)
    std_v11 = np.std(tempos_total_v11)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax1 = axes[0]
    labels_nms = ['Manual\n(Python)', 'Nativo\n(C++/CUDA)']
    medias_nms = [np.mean(tempos_nms_manual), np.mean(tempos_nms_torch)]
    erros_nms  = [std_nms_manual_intra, std_nms_torch_intra]
    cores_nms  = ['#1f77b4', '#ff7f0e']
    bars1 = ax1.bar(labels_nms, medias_nms, yerr=erros_nms, capsize=8,
                    color=cores_nms, alpha=0.8, edgecolor='black')
    ax1.set_title('NMS — Estabilidade Computacional\n(variância intra-imagem)',
                  fontsize=10, fontweight='bold')
    ax1.set_ylabel('Tempo (ms)', fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, err in zip(bars1, erros_nms):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + err * 1.2 + 0.01,
                 f'{yval:.3f}ms', ha='center', va='bottom', fontsize=9)

    ax2 = axes[1]
    medias_nms_inter = [np.mean(tempos_nms_manual_inter), np.mean(tempos_nms_torch_inter)]
    erros_nms_inter  = [np.std(tempos_nms_manual_inter), np.std(tempos_nms_torch_inter)]
    bars2 = ax2.bar(labels_nms, medias_nms_inter, yerr=erros_nms_inter, capsize=8,
                    color=cores_nms, alpha=0.8, edgecolor='black')
    ax2.set_title('NMS — Variação por Cena\n(variância inter-imagens)',
                  fontsize=10, fontweight='bold')
    ax2.set_ylabel('Tempo (ms)', fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, err in zip(bars2, erros_nms_inter):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + err * 1.2 + 0.01,
                 f'{yval:.3f}ms', ha='center', va='bottom', fontsize=9)

    ax3 = axes[2]
    labels_yolo = ['YOLOv3\nClássico', 'YOLOv11\nNano']
    medias_yolo = [np.mean(tempos_total_v3), np.mean(tempos_total_v11)]
    erros_yolo  = [std_v3, std_v11]
    cores_yolo  = ['#d62728', '#2ca02c']
    bars3 = ax3.bar(labels_yolo, medias_yolo, yerr=erros_yolo, capsize=8,
                    color=cores_yolo, alpha=0.8, edgecolor='black')
    ax3.set_title('Pipeline Completo: YOLOv3 vs YOLOv11\n(pré-proc + backbone + NMS único)',
                  fontsize=10, fontweight='bold')
    ax3.set_ylabel('Tempo (ms)', fontsize=11)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, err in zip(bars3, erros_yolo):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval + err * 0.5 + 0.5,
                 f'{yval:.2f}ms', ha='center', va='bottom', fontsize=9)

    plt.suptitle(
        'Análise de Tempo Computacional — Média ± Desvio Padrão\n'
        'Gráfico 1: estabilidade algorítmica  |  Gráficos 2–3: variação entre cenas',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    nome_arquivo_grafico = os.path.join(PASTA_ATUAL, 'benchmark_relatorio.png')
    plt.savefig(nome_arquivo_grafico, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo com sucesso em: {nome_arquivo_grafico}")
    # plt.show()