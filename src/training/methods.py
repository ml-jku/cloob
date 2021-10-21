import torch
import torch.nn as nn


def infoLOOB_loss(x, y, i, inv_tau):
    tau = 1 / inv_tau
    k = x @ y.T / tau
    positives = -torch.mean(torch.sum(k * i, dim=1))

    # For logsumexp the zero entries must be equal to a very large negative number
    large_neg = -1000.0
    arg_lse = k * torch.logical_not(i) + i * large_neg
    negatives = torch.mean(torch.logsumexp(arg_lse, dim=1))

    return tau * (positives + negatives)


def cloob(image_features, text_features, inv_tau, scale_hopfield):
    p_xx, p_yy, p_xy, p_yx = hopfield_retrieval(image_features, text_features, scale_hopfield)
    identity = torch.eye(p_xx.shape[1]) > 0.5
    i = identity.to(p_xx.device)
    loss_img = infoLOOB_loss(p_xx.T, p_xy.T, i, inv_tau=inv_tau)
    loss_txt = infoLOOB_loss(p_yy.T, p_yx.T, i, inv_tau=inv_tau)
    return loss_img + loss_txt


def clip(image_features, text_features, inv_tau, loss_fct_img, loss_fct_txt, args):
    logits_per_image = inv_tau * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    ground_truth = torch.arange(len(logits_per_image)).long()
    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
    loss_img = loss_fct_img(logits_per_image, ground_truth) / 2
    loss_txt = loss_fct_txt(logits_per_text, ground_truth) / 2
    return loss_img + loss_txt


def hopfield_retrieval(image_features, text_features, scale_hopfield):
    patterns_xx = hopfield(state_patterns=image_features, stored_patterns=image_features, scale_hopfield=scale_hopfield)
    patterns_yy = hopfield(state_patterns=text_features, stored_patterns=text_features, scale_hopfield=scale_hopfield)
    patterns_xy = hopfield(state_patterns=text_features, stored_patterns=image_features, scale_hopfield=scale_hopfield)
    patterns_yx = hopfield(state_patterns=image_features, stored_patterns=text_features, scale_hopfield=scale_hopfield)
    
    return patterns_xx, patterns_yy, patterns_xy, patterns_yx


def hopfield(state_patterns, stored_patterns, scale_hopfield):
    retrieved_patterns = stored_patterns.T @ nn.functional.softmax(
        scale_hopfield * stored_patterns @ state_patterns.t(), dim=0)
    # Column vectors -> dim=0 to normalize the column vectors
    retrieved_patterns = retrieved_patterns / retrieved_patterns.norm(dim=0, keepdim=True)
    return retrieved_patterns
