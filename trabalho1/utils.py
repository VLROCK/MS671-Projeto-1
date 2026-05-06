import numpy as np
import torch

def IoU():


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

        ordem = ordem[1:][not mascara_final]

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
