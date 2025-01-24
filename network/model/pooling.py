import utils
import torch
import torch.nn.functional as F

from torch_geometric.utils import scatter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getCommunityEmbedding(roiEmbeddings, batch_idx, dim, samples):
    if roiEmbeddings.shape[0] == samples:
        return roiEmbeddings
    else:
        commEmbedding = torch.zeros(samples, dim)
        batch_idx = sorted(set(batch_idx.tolist()))
        for re, idx in zip(roiEmbeddings, batch_idx):
            commEmbedding[idx] = re
        return commEmbedding.to(device)

# def communityPooling(x, batch, comm, corr, aggr="mean", rois=7, size=None):
#     xPooling = []
#     samples = torch.max(batch)+1
#     print("Comm = ", comm[:,0])
#     # print("Comm.shape = ", comm[:,0].unsqueeze(1))
#     print("X = ", x)
#     print("Out = ", x*comm[:,0].unsqueeze(1))
#     print(XY)
    

#     # get the community embedding
#     for roi in range(rois):
#         roi_x = x[comm == roi]
#         roi_batch = batch[comm == roi]

#         dim = -1 if isinstance(roi_x, torch.Tensor) and roi_x.dim() == 1 else -2
#         roi_pooling_x = scatter(roi_x, roi_batch, dim=dim, dim_size=size, reduce=aggr)
#         pooling_x = getCommunityEmbedding(roi_pooling_x, roi_batch, x.shape[-1], samples)
#         xPooling.append(pooling_x)
#     xPooling = torch.cat(xPooling, dim=1).reshape(-1, x.size(1))

#     # get the community batch
#     community_batch = torch.zeros(xPooling.shape[0], dtype=torch.long)

#     for idx in range(int(xPooling.shape[0]/rois)):
#         community_batch[idx*rois:(idx+1)*rois] = idx
    
#     # get the community_index and community_weight
#     community_index, community_weight = utils.getEdgeIdxAttr(corr)
#     community_index = community_index.repeat(1, int(xPooling.shape[0]/rois))
#     community_weight = community_weight.repeat(int(xPooling.shape[0]/rois))
#     return xPooling, community_index, community_weight, community_batch.to(device)


def communityPooling(x, batch, comm, corr, aggr="mean", rois=7, size=None):
    xPooling = []

    # get the community embedding
    for idx in range(comm.shape[-1]):
        roi_x = x*comm[:,idx].unsqueeze(1)

        dim = -1 if isinstance(roi_x, torch.Tensor) and roi_x.dim() == 1 else -2
        roi_pooling_x = scatter(roi_x, batch, dim=dim, dim_size=size, reduce=aggr)
        xPooling.append(roi_pooling_x)
    xPooling = torch.cat(xPooling, dim=1).reshape(-1, x.size(1))

    # get the community batch
    community_batch = torch.zeros(xPooling.shape[0], dtype=torch.long)

    for idx in range(int(xPooling.shape[0]/rois)):
        community_batch[idx*rois:(idx+1)*rois] = idx
    
    # get the community_index and community_weight
    community_index, community_weight = utils.getEdgeIdxAttr(corr)
    community_index = community_index.repeat(1, int(xPooling.shape[0]/rois))
    community_weight = community_weight.repeat(int(xPooling.shape[0]/rois))
    return xPooling, community_index, community_weight, community_batch.to(device)





