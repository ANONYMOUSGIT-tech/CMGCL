import torch

from torch import Tensor
from torch_geometric.utils import add_self_loops, add_remaining_self_loops, scatter

from torch_geometric.nn import SignedConv, GCNConv
from torch_geometric.typing import Adj, PairTensor, OptTensor, Union, Optional
from torch_geometric.utils.num_nodes import maybe_num_nodes


def signed_gcn_norm(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,):

    fill_value = 2. if improved else 1.

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    signedMX = torch.ones_like(deg)
    signedMX[torch.nonzero(deg < 0).squeeze()] = -1

    deg_inv_sqrt =  signedMX * deg.abs().pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class WeightedSignedConv(SignedConv):
    """
    WeightedSignedConv
    It is weighted gnn based on the SignedConvs

    parameters:
    ---------------
        in_channel: int
            The input channel
        out_channel: int
            The output channel
        first_aggr: bool
            The flag parameters. Is it the first layer?
    """

    def forward(self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_weight: OptTensor) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)
        
        pos_edge_index = torch.nonzero(edge_weight>0).squeeze()
        neg_edge_index = torch.nonzero(edge_weight<0).squeeze()

        if self.first_aggr:
            out_pos = self.propagate(edge_index[:,pos_edge_index], x=x, edge_weight=edge_weight[pos_edge_index])
            out_pos = self.lin_pos_l(out_pos)
            out_pos = out_pos + self.lin_pos_r(x[1])

            out_neg = self.propagate(edge_index[:,neg_edge_index], x=x, edge_weight=edge_weight[neg_edge_index])
            out_neg = self.lin_neg_l(out_neg)
            out_neg = out_neg + self.lin_neg_r(x[1])

            return torch.cat([out_pos, out_neg], dim=-1)
        
        else:
            F_in = self.in_channels

            out_pos1 = self.propagate(edge_index[:,pos_edge_index],
                                      x=(x[0][..., :F_in], x[1][..., :F_in]),
                                      edge_weight=edge_weight[pos_edge_index])
            out_pos2 = self.propagate(edge_index[:,neg_edge_index],
                                      x=(x[0][..., F_in:], x[1][..., F_in:]),
                                      edge_weight=edge_weight[neg_edge_index])
            out_pos = torch.cat([out_pos1, out_pos2], dim=-1)
            out_pos = self.lin_pos_l(out_pos)
            out_pos = out_pos + self.lin_pos_r(x[1][..., :F_in])

            out_neg1 = self.propagate(edge_index[:,pos_edge_index],
                                      x=(x[0][..., F_in:], x[1][..., F_in:]),
                                      edge_weight=edge_weight[pos_edge_index])
            out_neg2 = self.propagate(edge_index[:,neg_edge_index],
                                      x=(x[0][..., :F_in], x[1][..., :F_in]),
                                      edge_weight=edge_weight[neg_edge_index])
            out_neg = torch.cat([out_neg1, out_neg2], dim=-1)
            out_neg = self.lin_neg_l(out_neg)
            out_neg = out_neg + self.lin_neg_r(x[1][..., F_in:])

            return torch.cat([out_pos, out_neg], dim=-1)

class SignedGCNConv(GCNConv):
    """
    SignedGCNConv
    It is weighted gnn based on the GCN

    parameters:
    ---------------
        in_channel: int
            The input channel
        out_channel: int
            The output channel
    """
     
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
          
        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")
        
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = signed_gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
        
        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

