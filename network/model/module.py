import tqdm
import utils
import torch
import network.function as netF
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool

from network.model import convs
from network.model import pooling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if isinstance(m, GCNConv):
        torch.nn.init.xavier_uniform_(m.lin.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class GCNBlock(torch.nn.Module):
    """
    GCN Block
    It is a GCN Block. 
    For each block, it has a GCN + activation + dropout(depends on parameter).
    
    parameters:
    ---------------
    num_features: int
        The dimension of input features
    dim: int
        The dimension of output features
    activation: torch.nn.Module, optional
        THe activation function. Default is torch.nn.ReLU()
    dropout: float, optional
        The dropout rate. Default is None
    """
    def __init__(self, num_features, dim, activation=torch.nn.ReLU()):
        super(GCNBlock, self).__init__()
        self.conv = convs.WeightedSignedConv(num_features, dim, first_aggr=True) # My GCN with Signed
        self.activation = activation

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_weight=edge_attr)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Projector(torch.nn.Module):
    """
    Projector
    It is a projector layer.
    parameters:
    ---------------
    num_hidden: int
        The dimension of input features
    num_proj_hidden: int
        The dimension of output features
    """
    def __init__(self,num_hidden,num_proj_hidden):
        super(Projector, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Linear(num_hidden,num_proj_hidden),
            torch.nn.PReLU(),
            torch.nn.Linear(num_proj_hidden,num_hidden)
        )
    def forward(self,x):
        x = self.conv(x)
        return F.normalize(x)

class GraphNet(torch.nn.Module):
    """
    GraphNet
    It is a graph encoder using GCN Blocks.

    parameters:
    ---------------
    num_features: int
        The dimension of input features
    dims: list of int
        The dimension of hidden and output features
    activation: str, optional
        The activation function. Default is "relu"
    dropout: float, optional
        The dropout rate. Default is None
    """
    def __init__(self, num_features, dims, activation="relu"):
        super(GraphNet, self).__init__()
        self.activation = {"leakyrelu":torch.nn.LeakyReLU()}
        self.convs = torch.nn.ModuleList()

        for dim in dims[:-1]:
            self.convs.append(GCNBlock(num_features, dim, self.activation[activation]))
            num_features = dim*2
        self.convs.append(GCNBlock(num_features, dims[-1], None)) # My GCN with Signed
    
    def forward(self, data, comm=None, community_weight=None, num_comm=7):
        """
        if comm is None:
            it is used to train the community
        else:
            it is used to train the contrastive model
        """
        if comm is None:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            res = []
            for conv in self.convs:
                x = conv(x, edge_index, edge_attr)
                res.append(x)
            return torch.concat(res,dim=1)
        else:
            res = []
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

            xNode = self.convs[0](x, edge_index, edge_attr)

            xPooling, community_index, community_attr, community_batch = pooling.communityPooling(xNode, batch, comm, community_weight, rois=num_comm)

            out = self.convs[1](xPooling, community_index, community_attr)
            
            res = [global_mean_pool(out, community_batch), global_mean_pool(xNode, batch)]
            return torch.concat(res,dim=1), xPooling, xNode

class CommunityContrast(torch.nn.Module):

    def __init__(self, num_features, dims, tau, activation="relu", num_comm=7, gtau=0.1):
        super(CommunityContrast, self).__init__()
        self.tau = tau
        self.roi = num_features
        self.num_comm = num_comm
        self.gtau = gtau

        self.commNet = GraphNet(num_features, dims, activation)
        self.centroid = torch.nn.Parameter(torch.randn(num_comm, dims[-1]*4))
        torch.nn.init.xavier_uniform_(self.centroid.data, gain=5/3)
    

        self.contNet = GraphNet(num_features, dims, activation)
        self.proj_graph = Projector(dims[-1]*4, dims[-1]*2)
        self.proj_community = Projector(dims[-1]*2, dims[-1])
        self.proj_node = Projector(dims[-1]*2, dims[-1])
    
    def getCommunity(self, graph, iteration):
        centroid = self.centroid.detach()
        gtau = self.gtau + (1-self.gtau) * iteration
        comm = netF.cosineSimilarity(graph, F.normalize(centroid), self.tau, gtau=gtau, gumbel=True)
        return comm, torch.argmax(comm, dim=1)
        # return torch.argmax(comm, dim=1)
    
    def getCommunityWeight(self):
        """
        Computing the correlation of each community

        Parameters
        --------------
        centroid: torch.Tensor
            the embedding of each community centroid
        """
        centroid = self.centroid.detach()
        corr = centroid@centroid.T
        corr.abs_().fill_diagonal_(0)
        row_sums = corr.sum(dim=1)
        community_weight = corr/torch.sqrt(row_sums[:,None]*row_sums[None,:])
        return community_weight

    def forward(self, data, iteration):
        # Community
        community_node = self.commNet(data)
        comm, comm_softclass = self.getCommunity(community_node, iteration)
        community_weight = self.getCommunityWeight()

        # Get View
        view1, view2 = netF.getView(data, comm_softclass, community_weight)
        
        # # Graph
        gEmbedding_1, cEmbedding_1, nEmbedding1 = self.contNet(view1, comm, community_weight, num_comm=self.num_comm)
        gEmbedding_2, cEmbedding_2, nEmbedding2 = self.contNet(view2, comm, community_weight, num_comm=self.num_comm)
        return community_node, gEmbedding_1, cEmbedding_1, nEmbedding1, gEmbedding_2, cEmbedding_2, nEmbedding2
    
    def getEmbedding(self, data):
        _, out, _, _, _, _, _ = self(data,1)
        return out

    
    def projection(self, graphs, communities, nodes):
        return self.proj_graph(graphs), self.proj_community(communities), self.proj_node(nodes)
    
    def semi_loss(self, gh1, gh2, type="graph"):
        sim12 = netF.cosineSimilarity(gh1, gh2, stau=self.tau)
        sim21 = netF.cosineSimilarity(gh2, gh1, stau=self.tau)

        if type == "community":
            blocks = [torch.ones((self.num_comm, self.num_comm)) for _ in range(sim12.shape[0]//self.num_comm)]
            mask_diag = torch.block_diag(*blocks).to(device)
            sim12 = sim12 * mask_diag
            sim21 = sim21 * mask_diag
                
            
        elif type == "node":
            blocks = [torch.ones((self.roi, self.roi)) for _ in range(sim12.shape[0]//self.roi)]
            mask_diag = torch.block_diag(*blocks).to(device)
            sim12 = sim12 * mask_diag
            sim21 = sim21 * mask_diag

        loss12 = -torch.log(sim12.diag())+torch.log(sim12.sum(dim=1))
        loss21 = -torch.log(sim21.diag())+torch.log(sim21.sum(dim=1))

        loss = (loss12+loss21) / 2
        
        return loss.mean()
    
    def loss(self, data, community_node, gEmbedding_1, cEmbedding_1, nEmbedding1, gEmbedding_2, cEmbedding_2, nEmbedding2, iteration):
        gtau = self.gtau + (1-self.gtau) * iteration
        cosSim = netF.cosineSimilarity(community_node, F.normalize(self.centroid), self.tau, gtau=gtau, gumbel=True)
        loss_modularity = netF.modularity(data, cosSim)

        gh1, ch1, nh1 = self.projection(gEmbedding_1, cEmbedding_1, nEmbedding1)
        gh2, ch2, nh2 = self.projection(gEmbedding_2, cEmbedding_2, nEmbedding2)
        loss_graph = self.semi_loss(gh1, gh2, type="graph")
        loss_community = self.semi_loss(ch1, ch2, type="community")
        loss_node = self.semi_loss(nh1, nh2, type="node")

        return loss_modularity + loss_graph + loss_node + loss_community

