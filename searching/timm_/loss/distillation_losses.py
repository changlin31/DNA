import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['KLCosineSimilarity', 'L1Triplet', 'L2Triplet',
           'ContrastiveLoss', 'RKDDistance', 'RKDAngle', 'HardDarkRank', 'AttentionTransfer', 'cosine_similarity_loss']

class KLCosineSimilarity(nn.Module):
    r"""
    KL divergence between two distribution which is represented by cosine similarity loss between
    """

    def __init__(self):
        super(KLCosineSimilarity, self).__init__()

    def forward(self, x, target):
        return cosine_similarity_loss(x, target)


class _Triplet(nn.Module):
    def __init__(self, p=2, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin

        # update distance function accordingly
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p == 2))

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        loss = F.triplet_margin_loss(anchor_embed, positive_embed, negative_embed,
                                     margin=self.margin, p=self.p, reduction='none')

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class L2Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=2, margin=margin, sampler=sampler)

class L1Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=1, margin=margin, sampler=sampler)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__()
        self.margin = margin
        self.sampler = sampler

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        pos_loss = (F.pairwise_distance(anchor_embed, positive_embed, p=2)).pow(2)
        neg_loss = (self.margin - F.pairwise_distance(anchor_embed, negative_embed,
                                                      p=2)).clamp(min=0).pow(2)

        loss = torch.cat((pos_loss, neg_loss))
        return loss.mean()

class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:,
                      1:(self.permute_len + 1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack(
            [torch.logsumexp(ordered_student[:, i:], dim=1) for i in
             range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss

class FitNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.transform = nn.Conv2d(in_feature, out_feature, 1, bias=False)
        self.transform.weight.data.uniform_(-0.005, 0.005)

    def forward(self, student, teacher):
        if student.dim() == 2:
            student = student.unsqueeze(2).unsqueeze(3)
            teacher = teacher.unsqueeze(2).unsqueeze(3)

        return (self.transform(student) - teacher).pow(2).mean()

class AttentionTransfer(nn.Module):
    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(
                teacher.pow(2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()

class RKDAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C
        student = student.view(student.size(0), -1)
        teacher = teacher.view(teacher.size(0), -1)
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss

class RKDDistance(nn.Module):
    def forward(self, student, teacher):
        student = student.view(student.size(0), -1)
        teacher = teacher.view(teacher.size(0), -1)
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss


class L2CLLoss(nn.Module):
    def forward(self, student, teacher):
        return F.mse_loss(student, teacher)/teacher.var()


class FSPLoss(nn.Module):
    def forward(self, student_last, student, teacher_last, teacher):
        pass


def cosine_similarity_loss(output, target, eps=0.0000001):

    output_net = output.view(output.size(0), -1)
    target_net = target.view(target.size(0), -1)
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1,
                                                    keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1,
                                                      keepdim=True)

    # Calculate the KL-divergence
    loss = torch.sum(target_similarity * torch.log(
        (target_similarity + eps) / (model_similarity + eps)))

    return loss


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res