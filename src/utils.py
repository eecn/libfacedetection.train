#-*- coding:utf-8 -*-

import torch
import numpy as np


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 0:2] - boxes[:, 2:4]/2,     # xmin, ymin
                     boxes[:, 0:2] + boxes[:, 2:4]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(((boxes[:, 2:4] + boxes[:, 0:2])/2,  # cx, cy
                     boxes[:, 2:4] - boxes[:, 0:2]), 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:4].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, 0:2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 0:2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, 0:2], b[:, 0:2])
    rb = np.minimum(a[:, np.newaxis, 2:4], b[:, 2:4])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:4] - a[:, 0:2], axis=1)
    area_b = np.prod(b[:, 2:4] - b[:, 0:2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)

# intersection over foreground
def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, 0:2], b[:, 0:2])
    rb = np.minimum(a[:, np.newaxis, 2:4], b[:, 2:4])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:4] - a[:, 0:2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)

'''
根据图像的真实gt边框与生成的anchor进行匹配
1. 每一个gt匹配一个最大的anchor
2. 每一个anchor根据iou和阈值匹配gt
3. 根据匹配情况给anchor标记类别和偏移量
'''
def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return torch.zeros((1, priors.shape[0]))


    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,14]
    conf = labels[best_truth_idx]          # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,14] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior

    return best_truth_overlap

'''
计算前向推导预测结果的anchor和landmark与真实坐标之间的偏差
'''
def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes and landmarks (tensor), Shape: [num_priors, 14]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, 0:2] + matched[:, 2:4])/2 - priors[:, 0:2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:4])
    # match wh / prior wh
    g_wh = (matched[:, 2:4] - matched[:, 0:2]) / priors[:, 2:4]
    g_wh = torch.log(g_wh) / variances[1]

    # landmarks
    g_xy1 = (matched[:, 4:6] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    g_xy2 = (matched[:, 6:8] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    g_xy3 = (matched[:, 8:10] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    g_xy4 = (matched[:, 10:12] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    g_xy5 = (matched[:, 12:14] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])

    # return target for loss
    return torch.cat([g_cxcy, g_wh, g_xy1, g_xy2, g_xy3, g_xy4, g_xy5], 1)  # [num_priors,14]

'''
由预测的位置偏差、anchor 解码边界框和关键点的预测
'''
# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,14]  # 4-->14包含五个关键点
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    #print(priors[:, 0:2].shape)
    #print(loc[:, 0:2].shape)
    #print(priors[:, 2:4].shape)
    #print(loc[:, 0:2]* variances[0] * priors[:, 2:4])
    boxes = torch.cat((
        priors[:, 0:2] + loc[:, 0:2] * variances[0] * priors[:, 2:4],
        priors[:, 2:4] * torch.exp(loc[:, 2:4] * variances[1]),
        priors[:, 0:2] + loc[:, 4:6] * variances[0] * priors[:, 2:4],
        priors[:, 0:2] + loc[:, 6:8] * variances[0] * priors[:, 2:4],
        priors[:, 0:2] + loc[:, 8:10] * variances[0] * priors[:, 2:4],
        priors[:, 0:2] + loc[:, 10:12] * variances[0] * priors[:, 2:4],
        priors[:, 0:2] + loc[:, 12:14] * variances[0] * priors[:, 2:4]), 1)
    boxes[:, 0:2] -= boxes[:, 2:4] / 2
    boxes[:, 2:4] += boxes[:, 0:2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


'''
Log-softmax
1. softmax函数分子分母同乘或除非零数，分式值不变。为了防止向上溢出。指数上减去输入信号中的最大值，进行溢出抑制。
2. 减去最大值则最大输入为0，其余都为负数。可能存在向下溢出的情况。故对其使用log函数将除法转化为减法，减少下溢出风险，减少计算量，梯度优化更好。
这里只计算分母了
'''

if __name__=="__main__":
    # log_sum——exp test
    x = torch.randn((2,20),dtype=torch.float32)
    res = log_sum_exp(x)
    print(res)

    # decode test
    loc = torch.randn(1, 4385, 14)
    prior = torch.randn(1, 4385, 4)
    #print(loc.shape)
    #print(torch.squeeze(loc, axis=0).shape)
    dets = decode(torch.squeeze(loc, axis=0),torch.squeeze(prior, axis=0),[0.1,0.2])
    print(dets.shape)

    # encode test
    matchs = torch.randn(1, 4385, 4)
    priors = torch.randn(1, 4385, 4)
    var = encode(matchs, priors, [0.1,0.2])
    print(var.shape)

    # nms test
    boxes = torch.randn(1, 4385, 4)
    scores = torch.randn(1, 4385)
    keep, count = nms(torch.squeeze(boxes, axis=0),torch.squeeze(scores, axis=0), overlap=0.5, top_k=200)
    print(keep.shape)
    print(count)

    truths = torch.randn(1, 20, 4385)
    priors2 = torch.randn(1, 4385, 4)
    variances = torch.randn(1, 4385, 4)
    labels = torch.randn(1, 20)
    loc_t = torch.Tensor()
    conf_t = torch.Tensor()
    #best_truth_overlap = match(0.5, torch.squeeze(truths, axis=0), torch.squeeze(priors2, axis=0), [0.1,0.2], torch.squeeze(labels, axis=0), loc_t, conf_t, 0)



