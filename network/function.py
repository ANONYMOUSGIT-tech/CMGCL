import utils
import copy
import torch
import torch.nn.functional as F

from torch_geometric.data import Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def modularity_OUTDATE(graphs, communities, gamma=1):
    """
    Compute the modularity of a graph, according to the following formula:
    Q = \frac{1}{2m} * \sum_{ij}(A_{ij} - \frac{k_i*k_j}{2m}) * \delta(c_i, c_j))

    simplified formula:
    Q = \sum_{c=1}^{n}
       \left[ \frac{L_c}{m} - \gamma\left( \frac{k_c}{2m} \right) ^2 \right]

    parameters:
    ---------------
        graph: torch_geometric.data.Data
        community: list or iterable of set of nodes. These nodes set represent the communities of graph
        gamma: float, optional
            The resolution parameter. default is 1
    """
    outQ = 0
    cidx = 0
    roi = graphs.x.shape[1]
    def get_subgraph(graph, c):
        subgraph = torch.zeros(graph.x.shape[1], graph.x.shape[1])
        idx = c.nonzero().T[0]
        subgraph[tuple(zip(*[(row, col) for row in idx for col in idx]))] = 1
        return subgraph

    for i in range(len(graphs)):
        Q = 0
        m = sum(graphs[i].edge_attr)/2
        norm = 1 / (2*m)**2
        community = communities[cidx:cidx+roi].T
        cidx += roi
        

        for comm in community:
            subgraph = get_subgraph(graphs[i], comm)
            L_c = sum(comm@(subgraph*graphs[i].x))/2
            K_c = sum(comm@graphs[i].x)
            Q += L_c/m - gamma * K_c * K_c * norm
        outQ += torch.exp(-Q)
    return outQ/len(graphs)

def modularity(graphs, communities, gamma=1):
    """
    Compute the modularity of a graph, according to the following formula:
    Q = \frac{1}{2m} * \sum_{ij}(A_{ij} - \frac{k_i*k_j}{2m}) * \delta(c_i, c_j))

    simplified formula:
    Q = \sum_{c=1}^{n}
       \left[ \frac{L_c}{m} - \gamma\left( \frac{k_c}{2m} \right) ^2 \right]

    parameters:
    ---------------
        graph: torch_geometric.data.Data
        community: list or iterable of set of nodes. These nodes set represent the communities of graph
        gamma: float, optional
            The resolution parameter. default is 1
    """
    outQ = torch.zeros(1, device=communities.device)
    cidx = 0
    # communities = communitiess
    roi = graphs.x.shape[1]

    for i in range(len(graphs)):
        graph = graphs[i]
        m = graph.edge_attr.sum() / 2
        norm = 1 / (2 * m) ** 2
        community = communities[cidx : cidx + roi].T
        cidx += roi

        # Precompute degrees and edge sums
        degrees = graph.x.sum(dim=1)
        edge_weights = graph.x

        # Compute modularity for all communities
        Q = 0
        for comm in community:
            mask = comm.unsqueeze(1)
            subgraph = edge_weights * mask * mask.T
            L_c = subgraph.sum() / 2
            K_c = (degrees * comm).sum()
            Q += L_c / m - gamma * (K_c * K_c) * norm
        outQ = outQ + Q
    return -outQ / len(graphs)

def cosineSimilarity(x, y, stau=1, gtau=1, gumbel=False):
    """
    Calculate the cosine similarity between two matrices.
    """
    if gumbel:
        return F.gumbel_softmax(x@y.T/stau, tau=gtau, hard=False, dim=1)
    else:
        x_norm = x.norm(dim=1)
        y_norm = y.norm(dim=1)
        sim = torch.einsum('ik,jk->ij', x, y) / torch.einsum('i,j->ij', x_norm, y_norm)
        return torch.exp(sim / stau)

def delete_edges(edge_index, edge_attr, comm, corr):
        """
        Deleting graph edge based on the weights
        For positive weight, higher values have a higher probability of being kept.
        For negative weight, lower values have a higher probability of being kept.

        parameters:
        ---------------
        edge_index: torch.Tensor
            A 2-D tensor where the first row are the source nodes and the second row are the target nodes
        edge_attr: torch.Tensor
            A 1-D tensor containing the weight for each edge
        """
        negative_mask = edge_attr < 0
        positive_mask = edge_attr >= 0

        deletion_prob = torch.zeros_like(edge_attr)

        # Step 1: Compute deletion probabilities for negative and positive weights
        if negative_mask.any():
            min_negative_weight = torch.min(edge_attr[negative_mask])
            max_negative_weight = torch.max(edge_attr[negative_mask])
            range_negative = max_negative_weight - min_negative_weight
            if range_negative > 0:
                norm_negative_weight = (edge_attr[negative_mask] - min_negative_weight) / range_negative
                deletion_prob[negative_mask] = 1 - norm_negative_weight  # Favor smaller (more negative) values

        if positive_mask.any():
            min_positive_weight = torch.min(edge_attr[positive_mask])
            max_positive_weight = torch.max(edge_attr[positive_mask])
            range_positive = max_positive_weight - min_positive_weight
            if range_positive > 0:
                norm_positive_weight = (edge_attr[positive_mask] - min_positive_weight) / range_positive
                deletion_prob[positive_mask] = norm_positive_weight  # Favor larger values

        # Step 2: Adjust probabilities for inter-community edges
        source_comm = comm[edge_index[0]]
        target_comm = comm[edge_index[1]]
        inter_comm_mask = source_comm != target_comm

        if inter_comm_mask.any():
            comm_corr_values = corr[source_comm[inter_comm_mask], target_comm[inter_comm_mask]]
            deletion_prob[inter_comm_mask] *= comm_corr_values

        # Step 3: Randomized edge deletion
        random_value = torch.rand_like(edge_attr)
        kept_mask = random_value < deletion_prob

        return edge_index[:, kept_mask], edge_attr[kept_mask]

def symGraph(edge_index, edge_attr, rois):
    """
    Get the augmented edge_index and edge_attr, then generating the symmetric matrix

    Parameters
    -------------
    edge_index: torch.Tensor
        the first row is the source nodes and the second row is the target nodes
    edge_attr: torch.Tensor
        the weight of each edge
    rois: int
        the number of nodes 
    """

    # Create the adjacency matrix directly using sparse operations
    newGraph = torch.zeros((rois, rois)).to(device)
    newGraph[edge_index[0], edge_index[1]] = edge_attr

    symNewGraph = torch.maximum(newGraph, newGraph.T)
    sym_edge_index, sym_edge_attr = utils.getEdgeIdxAttr(symNewGraph)
    return sym_edge_index, sym_edge_attr


def getView(graphs, comm, community_weight):
    """
    get two view of input graph

    Parameters
    -------------------
    graphs: DataBatch
        the batch of graph 
    """
    view1, view2 = graphs.to_data_list(), graphs.to_data_list()
    
    def getAugEdgeIndexAttr(graph):
        augGraph = copy.deepcopy(graph)
        aug_edge_index, aug_edge_attr = delete_edges(augGraph.edge_index, augGraph.edge_attr, comm, community_weight)
        return symGraph(aug_edge_index, aug_edge_attr, augGraph.x.shape[0])

    for i in range(len(graphs)):
        view1[i].edge_index, view1[i].edge_attr = getAugEdgeIndexAttr(graphs[i])
        view2[i].edge_index, view2[i].edge_attr = getAugEdgeIndexAttr(graphs[i])

    return Batch.from_data_list(view1).to(device), Batch.from_data_list(view2).to(device)
