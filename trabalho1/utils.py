import numpy as np
import torch
from torchvision import ops


def IoU(box, boxes_restantes):

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
    intersecao_y_max = np.maximum(y_max, Y_max)
    intersecao_x_max = np.maximum(x_max, X_max)
    intersecao_y_min = np.maximum(y_min, Y_min)
    intersecao_x_min = np.maximum(x_min, X_min)
    
    intersecao_h = np.clip(intersecao_y_max - intersecao_y_min, 0, None)
    intersecao_w = np.clip(intersecao_x_max - intersecao_x_min, 0, None)

    area_intersecao = intersecao_h * intersecao_w

    #calcula area total = caixa + area_caixas - intersecao --> vetor com a todas as areas totais
    area_uniao = area_caixa + area_caixas_restantes - area_intersecao

    #calcula razao = area intersecao(vetor) / area total (vetor)

    return area_intersecao/area_uniao




def manual_nms(boxes, scores, iou_threshold):
    pass