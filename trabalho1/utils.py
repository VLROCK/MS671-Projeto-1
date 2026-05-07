import numpy as np
import torch
from torchvision import ops


def IoU(box, boxes_restantes):
    """
    Função que calcula um vetor IoU para uma caixa em relação a um vetor de outras caixas.

    box: Vetor de tamanho (1,4), no formato [y_min, x_min, y_max, x_max]
    boxes_restantes: Vetor com as outras caixas para serem comparadas com box.

    return: Vetor com os valores de IoU
    """

    y_min, x_min, y_max, x_max = box[0], box[1], box[2], box[3]

    Y_min = boxes_restantes[:, 0]
    X_min = boxes_restantes[:, 1]
    Y_max = boxes_restantes[:, 2]
    X_max = boxes_restantes[:, 3]

    #calcula area da caixa 
    area_caixa = (x_max - x_min) * (y_max - y_min)


    #calcula area de todas as caixas restantes
    area_caixas_restantes = (X_max - X_min) * (Y_max - Y_min)

    #calcula area das intersecoes --> precisa retornar um vetor
    intersecao_y_max = np.minimum(y_max, Y_max)
    intersecao_x_max = np.minimum(x_max, X_max)
    intersecao_y_min = np.maximum(y_min, Y_min)
    intersecao_x_min = np.maximum(x_min, X_min)
    
    intersecao_h = np.clip(intersecao_y_max - intersecao_y_min, 0, None)
    intersecao_w = np.clip(intersecao_x_max - intersecao_x_min, 0, None)

    area_intersecao = intersecao_h * intersecao_w

    #calcula area total = caixa + area_caixas - intersecao --> vetor com a todas as areas totais
    area_uniao = area_caixa + area_caixas_restantes - area_intersecao

    #calcula razao = area intersecao(vetor) / area total (vetor)

    return area_intersecao/area_uniao



def manual_nms(boxes, scores, classes, iou_threshold):
    """
    Non-Maximum Suppression manual com separação por classes.
    
    boxes: Tensor de shape (N, 4) no formato [y_min, x_min, y_max, x_max]
    scores: Tensor de shape (N,) com as confianças
    classes: Tensor de shape (N,) com os IDs das classes
    iou_threshold: Float definindo o limite de sobreposição
    """

    keep = []

    ordem = torch.argsort(scores, descending=True)
    while len(ordem) > 0:
        idx = ordem[0].item()
        keep.append(idx)

        if len(ordem) == 1:
            break

        caixas_restantes = boxes[ordem[1:]]
        
        ious = IoU(boxes[idx], caixas_restantes)

        mascara_classe = (classes[ordem[1:]] == classes[idx])

        mascara_iou = (ious > iou_threshold)

        mascara_final = mascara_classe & mascara_iou

        ordem = ordem[1:][~mascara_final]

    return keep

## Código do manual_nms não vetorizado
# def manual_nms(boxes, scores, classes, iou_threshold):
#     boxes = boxes.cpu().numpy()
#     for i in range(len(boxes)):
#         for j in range(i+1, len(boxes)):
#             if classes[i] == classes[j]:
#                 if IoU(boxes[i], boxes[j]) > iou_threshold:
#                     if scores[i] > scores[j]:
#                         boxes[j] = [0, 0, 0, 0]
#                     else:
#                         boxes[i] = [0, 0, 0, 0]
#     keep = [i for i in range(len(boxes)) if not np.array_equal(boxes[i], [0, 0, 0, 0])]
#     return keep
