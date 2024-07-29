import torch
import torch.nn as nn
import torch.nn.functional as F


def triplet_loss_function(sk_p, im_p, im_n, args):
    triplet = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
        margin=args.MARGIN).cuda()

    loss = triplet(sk_p, im_p, im_n)

    return loss


def sdm_loss_function(sketch_features, image_features, sketch_pids, image_pids, logit_scale, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    assert sketch_features.shape[0] == image_features.shape[0] == sketch_pids.shape[0] == image_pids.shape[0]

    s2i_cosine_theta = sketch_features @ image_features.t()
    i2s_cosine_theta = s2i_cosine_theta.t()
    sketch_proj_image = logit_scale * s2i_cosine_theta
    image_proj_sketch = logit_scale * i2s_cosine_theta

    batch_size = sketch_features.shape[0]
    sketch_pids = sketch_pids.reshape((batch_size, 1))
    image_pids = image_pids.reshape((batch_size, 1))

    labels = torch.eq(sketch_pids, image_pids.T).float()
    labels_distribute = labels / (labels.sum(dim=1) + 1e-10)

    s2i_pred = F.softmax(sketch_proj_image, dim=1)
    s2i_loss = s2i_pred * (F.log_softmax(sketch_proj_image, dim=1) - torch.log(labels_distribute + epsilon))
    i2s_pred = F.softmax(image_proj_sketch, dim=1)
    i2s_loss = i2s_pred * (F.log_softmax(image_proj_sketch, dim=1) - torch.log(labels_distribute + epsilon))

    loss = (torch.mean(torch.sum(s2i_loss, dim=1))
            + torch.mean(torch.sum(i2s_loss, dim=1))
            ) / 2

    return loss
