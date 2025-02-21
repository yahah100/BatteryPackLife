"""
Loss functions for PyTorch.
"""
import torch
import torch as t
import torch.nn as nn
import numpy as np
import pdb
import torch.nn.functional as F
# class Battery_life_alignment_CL_loss(nn.Module):
#     '''
#     Contrastive learning loss for battery life prediction
#     '''
#     def __init__(self, args, cl_loss_weight=0.5, instance_tau=1, cluster_tau=1):
#         super(Battery_life_alignment_CL_loss, self).__init__()
#         self.args = args
#         self.pdist = nn.PairwiseDistance(p=2)
#         self.instance_tau = instance_tau
#         # self.cluster_tau = cluster_tau
#         self.cl_loss_weight = cl_loss_weight
    


#     def forward(self, features, cl_embed, center_vectors):
#         '''
#         features: [2*N, L, D]
#         cl_embed: [2*N, d_ff]
#         center_vectors: [token_num, D]
#         '''
#         N, L, D = features.shape[0]//2, features.shape[1], features.shape[-1]
#         token_num = center_vectors.shape[0]
#         # raw_features = features.clone()
#         # center_vectors = F.normalize(center_vectors, dim=-1, p=2)
#         # features = F.normalize(features, dim=-1, p=2)
#         aug_features = features[N:]
#         anchor_features = features[:N]
#         # alignment loss
#         # cos_similarities = torch.matmul(anchor_features, center_vectors.transpose(0,1)) # [N, L, token_num]
#         # cos_similarities = cos_similarities.reshape(N, -1) # [N, L*toekn_num]
#         # cos_similarities = torch.mean(cos_similarities, dim=1) # [N]
#         # alignment_loss = - torch.mean(cos_similarities) # 1
#         center_vectors = center_vectors.unsqueeze(0)
#         # center_vectors = torch.repeat_interleave(center_vectors, repeats=N, dim=0) # [N, token_num, D]
#         center_vectors = center_vectors.expand(N, -1, -1)
#         tmp_anchor_features = anchor_features.unsqueeze(2) # [N, L, 1, D]
#         tmp_anchor_features = tmp_anchor_features.expand(-1, -1, token_num, -1) # [N, L, token_num, D]
#         tmp_center_vectors = center_vectors.unsqueeze(1) # [N, 1, token_num, D]
#         tmp_center_vectors = tmp_center_vectors.expand(-1, L, -1, -1) # [N, L, token_num, D]



#         sim = - torch.norm(tmp_anchor_features - tmp_center_vectors, p=2, dim=-1) # [N, L, token_num]
#         sim = sim.reshape(N, -1) # [N, L*toekn_num]
#         sim = torch.mean(sim, dim=1) # [N]
#         alignment_loss = - torch.mean(sim) # 1

#         # use contrastive learning to avoid collapse
#         # Sim between anchor sample and postive sample
#         # pos_sim = torch.matmul(anchor_features, aug_features.transpose(1,2)) # [N, L, L]
#         # pos_sim = pos_sim.reshape(N, -1) # [N, L*L]
#         # pos_sim = torch.mean(pos_sim, dim=1) # [N]
#         # pos_sim = torch.exp(pos_sim/self.instance_tau) # [N]
#         aug_features = cl_embed[N:] # [N, d_ff]
#         anchor_features = cl_embed[:N]
#         D = cl_embed.shape[-1]

#         pos_sim = - self.pdist(anchor_features, aug_features) # [N]
#         pos_sim = torch.exp(pos_sim/self.instance_tau) # [N]

#         # Sim between anchor and negative sample
#         # neg_sim = torch.matmul(features.reshape(-1, D), features.reshape(-1, D).transpose(0,1)) # [2*N*L, 2*N*L]
#         # neg_sim = torch.mean(neg_sim, dim=1) # [2*N*L]
#         # neg_sim = neg_sim.reshape(2*N, -1) # [2N, L]
#         # neg_sim = torch.mean(neg_sim, dim=1) # [2N]
#         # neg_sim = torch.exp(neg_sim/self.instance_tau)
#         # neg_sim = torch.sum(neg_sim) # 1
#         tmp_features = cl_embed[:, None].expand(-1, 2*N, -1) # [N, N, D]


#         neg_sim = - torch.norm(tmp_features - tmp_features.transpose(0,1), dim=2, p=2) # [N, N]
#         neg_sim = torch.mean(neg_sim, dim=1) # [N]
#         neg_sim = torch.exp(neg_sim/self.instance_tau) # [N]
#         neg_sim = torch.sum(neg_sim) # 1

#         cl_loss = - torch.mean(torch.log(pos_sim / neg_sim))

#         loss = alignment_loss + self.cl_loss_weight * cl_loss

#         return loss, alignment_loss, cl_loss  


class Battery_life_alignment_CL_loss_C0241029(nn.Module):
    '''
    Contrastive learning loss for battery life prediction
    '''
    def __init__(self, args, dist_threshold, DG_weight, cl_loss_weight=0.01, instance_tau=1, cluster_tau=1):
        super(Battery_life_alignment_CL_loss, self).__init__()
        self.args = args
        self.dist_threshold = dist_threshold
        self.DG_weight = DG_weight
        self.pdist = nn.PairwiseDistance(p=2)
        self.instance_tau = instance_tau
        # self.cluster_tau = cluster_tau
        self.cl_loss_weight = cl_loss_weight
    


    def forward(self, features, cl_embed, center_vectors, parametric_centers, class_labels, curve_attn_mask):
        '''
        features: [2*N, L, D]
        cl_embed: [2*N, d_ff]
        center_vectors: [token_num, D] the center vectors from the text modality
        parametric_centers: [num_class, D] the centers for each life class
        class_labels: [N] the life class labels
        curve_attn_mask: [2N, L] 0 means the position is masked
        '''
        N, L, D = features.shape[0]//2, features.shape[1], features.shape[-1]
        curve_attn_mask = curve_attn_mask[:N]
        token_num = center_vectors.shape[0]
        # raw_features = features.clone()
        # center_vectors = F.normalize(center_vectors, dim=-1, p=2)
        # features = F.normalize(features, dim=-1, p=2)
        aug_features = features[N:]
        anchor_features = features[:N]
        # alignment loss
        # Align the modality between the pretrained token embeddings and learned token embeddings
        # The embeddings for the padding tokens are ignored since they are masked in attention
        # If DG is used, the parametric centers will also be aligned to the LLM space
        if self.args.use_align:
            center_vectors = center_vectors.unsqueeze(0)
            center_vectors = center_vectors.expand(N, -1, -1)
            tmp_anchor_features = anchor_features.unsqueeze(2) # [N, L, 1, D]
            if self.args.DG:
                tmp_parametric_centers = parametric_centers.unsqueeze(0) # [1, num_class, D]
                tmp_parametric_centers = tmp_parametric_centers.unsqueeze(2) # [1, num_class, 1, D]
                tmp_parametric_centers = tmp_parametric_centers.expand(N, -1, -1, -1) # [N, num_class, 1, D]
                tmp_anchor_features = torch.cat([tmp_anchor_features, tmp_parametric_centers], dim=1) # [N, L+num_class, 1, D]

            tmp_anchor_features = tmp_anchor_features.expand(-1, -1, token_num, -1) # [N, L, token_num, D]
            tmp_center_vectors = center_vectors.unsqueeze(1) # [N, 1, token_num, D]
            tmp_center_vectors = tmp_center_vectors.expand(-1, tmp_anchor_features.shape[1], -1, -1) # [N, L, token_num, D]

            dist = torch.norm(tmp_anchor_features - tmp_center_vectors, p=2, dim=-1) # [N, L, token_num]
            curve_attn_mask = curve_attn_mask.unsqueeze(-1).expand(-1, -1, token_num) # [N, L, token_num]
            if self.args.DG:
                tmp_parametric_centers = parametric_centers.unsqueeze(0) # [1, num_class, D]
                tmp_parametric_centers = tmp_parametric_centers.expand(N, -1, -1) # [N, num_class, D]
                parametric_center_masks = torch.ones_like(tmp_parametric_centers[:,:,0]) # [N, num_class]
                parametric_center_masks = parametric_center_masks.unsqueeze(-1).expand(-1,-1,token_num) # [N, num_class, token_num]
                curve_attn_mask = torch.cat([curve_attn_mask, parametric_center_masks], dim=1)

            dist = torch.where(curve_attn_mask==1, dist, torch.zeros_like(dist)) # [N, L, token_num]
            dist = dist.reshape(N, -1) # [N, L*toekn_num]
            curve_attn_mask = curve_attn_mask.reshape(N, -1)  # [N, L*toekn_num]
            dist = torch.sum(dist, dim=1) / torch.sum(curve_attn_mask, dim=1) # [N]
            alignment_loss = F.relu(dist-self.dist_threshold) # loss with a margin
            alignment_loss = torch.mean(alignment_loss) # 1

        # use contrastive learning 
        aug_features = cl_embed[N:] # [N, d_llm]
        anchor_features = cl_embed[:N]  # [N, d_llm]

        pos_sim = - self.pdist(anchor_features, aug_features) # [N]
        pos_sim = torch.exp(pos_sim/self.instance_tau) # [N]
        # Sim between anchor and negative sample
        tmp_features = cl_embed[:, None].expand(-1, N, -1) # [2N, N, D]
        tmp_anchor_features = anchor_features[:,None].expand(-1, 2*N, -1) # [N, 2N, D]

        neg_sim = - torch.norm(tmp_anchor_features - tmp_features.transpose(0,1), dim=2, p=2) # [N, 2N]
        neg_sim = torch.exp(neg_sim/self.instance_tau) # [N, 2N]
        if not self.args.DG:
            neg_sim = torch.sum(neg_sim, dim=1) # [N]
            cl_loss = - torch.mean(torch.log(pos_sim / neg_sim))
            return_cl_loss = cl_loss
            DG_loss = 0
        else:
            # use DG
            # add domain generalization loss
            # positive samples
            anchor_features = cl_embed[:N] # [N, d_llm]
            selected_centers = torch.index_select(parametric_centers, dim=0, index=class_labels)
            DG_pos_sim = - self.pdist(anchor_features, selected_centers) # [N]
            DG_pos_sim = torch.exp(DG_pos_sim/self.instance_tau) # [N]

            # negative samples
            anchor_features = anchor_features[:,None].expand(-1,N,-1) # [N, N, d_llm]
            selected_centers = selected_centers[:,None].expand(-1,N,-1) # [N, N, d_llm]

            DG_neg_sim = - torch.norm(anchor_features - selected_centers.transpose(0,1), dim=2, p=2) # [N, N]
            DG_neg_sim = torch.exp(DG_neg_sim/self.instance_tau) # [N, N]
            neg_sim = torch.cat([neg_sim, DG_neg_sim], dim=1) # [N, 3N]
            neg_sim = torch.sum(neg_sim, dim=1) # N

            DG_term = torch.log(DG_pos_sim / neg_sim) # [N]
            CL_term = torch.log(pos_sim / neg_sim) # [N]

            DG_loss = - torch.mean(DG_term) # 1
            return_cl_loss = - torch.mean(CL_term) # 1
            cl_loss = - torch.mean(DG_term + CL_term)

        if self.args.use_align:
            loss = alignment_loss + self.cl_loss_weight * cl_loss
        else:
            loss = self.cl_loss_weight * cl_loss
            alignment_loss = 0

        return loss, alignment_loss, return_cl_loss, DG_loss
    
class Battery_life_alignment_CL_loss(nn.Module):
    '''
    Contrastive learning loss for battery life prediction
    '''
    def __init__(self, args, dist_threshold, DG_weight, max_mean, max_std, cl_loss_weight=0.01, instance_tau=1, cluster_tau=1):
        super(Battery_life_alignment_CL_loss, self).__init__()
        self.args = args
        self.dist_threshold = dist_threshold
        self.DG_weight = DG_weight
        self.pdist = nn.PairwiseDistance(p=2)
        self.instance_tau = instance_tau
        # self.cluster_tau = cluster_tau
        self.cl_loss_weight = cl_loss_weight
        self.max_mean = max_mean
        self.max_std = max_std
    


    def forward(self, features, cl_embed, center_vectors, parametric_centers, class_labels, curve_attn_mask):
        '''
        features: [2*N, L, D]
        cl_embed: [2*N, d_ff]
        center_vectors: [token_num, D] the center vectors from the text modality
        parametric_centers: [num_class, D] the centers for each life class
        class_labels: [N] the life class labels
        curve_attn_mask: [2N, L] 0 means the position is masked
        '''
        N, L, D = features.shape[0]//2, features.shape[1], features.shape[-1]
        curve_attn_mask = curve_attn_mask[:N]
        token_num = center_vectors.shape[0]
        # raw_features = features.clone()
        # center_vectors = F.normalize(center_vectors, dim=-1, p=2)
        # features = F.normalize(features, dim=-1, p=2)
        aug_features = features[N:]
        anchor_features = features[:N]
        # alignment loss
        # Align the modality between the pretrained token embeddings and learned token embeddings
        # The embeddings for the padding tokens are ignored since they are masked in attention
        # If DG is used, the parametric centers will also be aligned to the LLM space
        if self.args.use_align:
            center_vectors = center_vectors.unsqueeze(0)
            center_vectors = center_vectors.expand(N, -1, -1)
            tmp_anchor_features = anchor_features.unsqueeze(2) # [N, L, 1, D]
            if self.args.DG:
                tmp_parametric_centers = parametric_centers.unsqueeze(0) # [1, num_class, D]
                tmp_parametric_centers = tmp_parametric_centers.unsqueeze(2) # [1, num_class, 1, D]
                tmp_parametric_centers = tmp_parametric_centers.expand(N, -1, -1, -1) # [N, num_class, 1, D]
                tmp_anchor_features = torch.cat([tmp_anchor_features, tmp_parametric_centers], dim=1) # [N, L+num_class, 1, D]

            tmp_anchor_features = tmp_anchor_features.expand(-1, -1, token_num, -1) # [N, L, token_num, D]
            tmp_center_vectors = center_vectors.unsqueeze(1) # [N, 1, token_num, D]
            tmp_center_vectors = tmp_center_vectors.expand(-1, tmp_anchor_features.shape[1], -1, -1) # [N, L, token_num, D]
            # We only hope that the scale of two modality are aligned
            center_mean = torch.mean(tmp_center_vectors, dim=-1) # [N, L, token_num]
            anchor_mean = torch.mean(tmp_anchor_features, dim=-1) # [N, L, token_num]

            center_std = torch.std(tmp_center_vectors, dim=-1)
            anchor_std = torch.std(tmp_anchor_features, dim=-1)


            # dist = (anchor_mean-center_mean)*(anchor_mean-center_mean) # squared error w.r.t mean
            # dist += (anchor_std-center_std)*(anchor_std-center_std) # squared error w.r.t std
            mean_dist = F.relu(torch.abs(anchor_mean - center_mean) - self.max_mean)
            std_dist = F.relu(torch.abs(anchor_std - center_std) - self.max_std)
            dist = mean_dist + std_dist
            curve_attn_mask = curve_attn_mask.unsqueeze(-1).expand(-1,-1,token_num) # [N, L, token_num]
            if self.args.DG:
                tmp_parametric_centers = parametric_centers.unsqueeze(0) # [1, num_class, D]
                tmp_parametric_centers = tmp_parametric_centers.expand(N, -1, -1) # [N, num_class, D]
                parametric_center_masks = torch.ones_like(tmp_parametric_centers[:,:,0]) # [N, num_class]
                parametric_center_masks = parametric_center_masks.unsqueeze(-1).expand(-1,-1,token_num) # [N, num_class, token_num]
                curve_attn_mask = torch.cat([curve_attn_mask, parametric_center_masks], dim=1)

            dist = torch.where(curve_attn_mask==1, dist, torch.zeros_like(dist)) # [N, L, token_num]
            dist = dist.reshape(N, -1) # [N, L*toekn_num]
            curve_attn_mask = curve_attn_mask.reshape(N, -1)  # [N, L*toekn_num]
            dist = torch.sum(dist, dim=1) / torch.sum(curve_attn_mask, dim=1) # [N]
            # alignment_loss = F.relu(dist-self.dist_threshold) # loss with a margin
            alignment_loss = dist
            alignment_loss = torch.mean(alignment_loss) # 1

        # use contrastive learning 
        aug_features = cl_embed[N:] # [N, d_llm]
        anchor_features = cl_embed[:N]  # [N, d_llm]

        pos_sim = - self.pdist(anchor_features, aug_features) # [N]
        pos_sim = torch.exp(pos_sim/self.instance_tau) # [N]
        # Sim between anchor and negative sample
        tmp_features = cl_embed[:, None].expand(-1, N, -1) # [2N, N, D]
        tmp_anchor_features = anchor_features[:,None].expand(-1, 2*N, -1) # [N, 2N, D]

        neg_sim = - torch.norm(tmp_anchor_features - tmp_features.transpose(0,1), dim=2, p=2) # [N, 2N]
        neg_sim = torch.exp(neg_sim/self.instance_tau) # [N, 2N]
        if not self.args.DG:
            neg_sim = torch.sum(neg_sim, dim=1) # [N]
            cl_loss = - torch.mean(torch.log(pos_sim / neg_sim))
            return_cl_loss = cl_loss
            DG_loss = 0
        else:
            # use DG
            # add domain generalization loss
            # positive samples
            anchor_features = cl_embed[:N] # [N, d_llm]
            selected_centers = torch.index_select(parametric_centers, dim=0, index=class_labels)
            DG_pos_sim = - self.pdist(anchor_features, selected_centers) # [N]
            DG_pos_sim = torch.exp(DG_pos_sim/self.instance_tau) # [N]

            # negative samples
            anchor_features = anchor_features[:,None].expand(-1,N,-1) # [N, N, d_llm]
            selected_centers = selected_centers[:,None].expand(-1,N,-1) # [N, N, d_llm]

            DG_neg_sim = - torch.norm(anchor_features - selected_centers.transpose(0,1), dim=2, p=2) # [N, N]
            DG_neg_sim = torch.exp(DG_neg_sim/self.instance_tau) # [N, N]
            neg_sim = torch.cat([neg_sim, DG_neg_sim], dim=1) # [N, 3N]
            neg_sim = torch.sum(neg_sim, dim=1) # N

            DG_term = torch.log(DG_pos_sim / neg_sim) # [N]
            CL_term = torch.log(pos_sim / neg_sim) # [N]

            DG_loss = - torch.mean(DG_term) # 1
            return_cl_loss = - torch.mean(CL_term) # 1
            cl_loss = - torch.mean(DG_term + CL_term)

        if self.args.use_align:
            loss = alignment_loss + self.cl_loss_weight * cl_loss
        else:
            loss = self.cl_loss_weight * cl_loss
            alignment_loss = 0

        return loss, alignment_loss, return_cl_loss, DG_loss
    
class Battery_life_CL_loss(nn.Module):
    '''
    Contrastive learning loss for battery life prediction
    '''
    def __init__(self, cluster_cl_loss_weight=5, instance_tau=1, cluster_tau=1):
        super(Battery_life_CL_loss, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)
        self.instance_tau = instance_tau
        self.cluster_tau = cluster_tau
        self.cluster_cl_loss_weight = cluster_cl_loss_weight
    


    def forward(self, features, center_vectors, center_vectors_indices):
        '''
        features: [N, D]
        center_vectors_indices: [N]
        center_vectors: [center_num, D]
        '''
        N = features.shape[0]//2
        center_vectors = F.normalize(center_vectors, dim=1, p=2)
        features = F.normalize(features, dim=1, p=2)
        aug_features = features[features.shape[0]//2:] # [N, D]
        features = features[:features.shape[0]//2] # [N, D]

        # Instance-wise contrastive learning
        # Anchor with positive samples
        pos_dist = self.pdist(features, aug_features) / self.instance_tau # [N]
        pos_dist = torch.exp(pos_dist) # [N]

        # Anchor with negative samples
        sim_m = torch.norm(features[:, None]-features, dim=2, p=2) / self.instance_tau # [N, N]
        sim_m = torch.exp(sim_m) # [N, N]
        denominator = torch.sum(sim_m, dim=1) # [N]

        instance_wise_cl_loss = torch.mean(torch.log(pos_dist / denominator))

        # Cluster-wise constrastive learning
        # Anchor with its center

        # anchor_centers = center_vectors[center_vectors_indices] # [N, D]
        anchor_centers = torch.index_select(center_vectors, dim=0, index=center_vectors_indices)
        pos_sim = torch.cosine_similarity(features, anchor_centers, dim=1) / self.cluster_tau # [N]
        pos_sim = torch.exp(pos_sim)

        # Anchor with other centers
        features = F.normalize(features, p=2, dim=1) # [N,D]
        center_vectors = F.normalize(center_vectors, p=2, dim=1) # [center_num,D]
        sim_m = torch.mm(features, center_vectors.transpose(0,1)) # [N, center_num]
        sim_m = sim_m / self.cluster_tau  # [N, center_num]
        sim_m = torch.exp(sim_m)
        denominator = torch.sum(sim_m, dim=1) # [N]

        cluster_wise_cl_loss = - torch.mean(torch.log(pos_sim / denominator))


        cl_loss = instance_wise_cl_loss + self.cluster_cl_loss_weight * cluster_wise_cl_loss

        return cl_loss   
        
        
        
        
       

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
