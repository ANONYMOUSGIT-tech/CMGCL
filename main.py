import tqdm
import utils
import torch
import random
import argparse
import dataloader
import evaluation

import numpy as np

import network.model.build as build
from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(net, opt, dataLoader, iteration):
    net.train()
    ls = 0
    for data in dataLoader:
        data = data.to(device)
        opt.zero_grad()
        community_node, gEmbedding_1, cEmbedding_1, nEmbedding1, gEmbedding_2, cEmbedding_2, nEmbedding2 = net(data, iteration)
        loss = net.loss(data, community_node, gEmbedding_1, cEmbedding_1, nEmbedding1, gEmbedding_2, cEmbedding_2, nEmbedding2, iteration)
        loss.backward()
        opt.step()
        ls += loss.item()
    return ls

def eval(net, dataLoader):
    net.eval()
    x,y = [],[]
    for data in dataLoader:
        data = data.to(device)
        out = net.getEmbedding(data)
        y.append(data.y.detach().cpu().numpy())
        x.append(out.detach().cpu().numpy())

    embeds = np.concatenate(x, 0)
    labels = np.concatenate(y, 0)
    
    acc, f1, auc, spe = evaluation.mlp_evaluator(embeds, labels)
    return acc, f1, auc, spe


def train_eval(net, opt, dataLoader, epochs):

    pbar = tqdm.tqdm(range(epochs))
    bestLoss = 1e9
    checkPoint = net.state_dict()

    for epoch in pbar:
        pbar.set_description('Epoch %d...' % epoch)
        loss = train(net, opt, dataLoader, iteration=epoch/epochs)
        pbar.set_postfix(Loss=loss)
    
        if loss < bestLoss:
            checkPoint = net.state_dict()
            bestLoss = loss
    
    # Evaluating
    print(f"=== Result===")
    net.load_state_dict(checkPoint)
    acc, f1, auc, spe = eval(net, dataLoader)
    print(f"acc = {np.mean(acc):.4f}±{np.std(acc):.4f}, f1 = {np.mean(f1):.4f}±{np.std(f1):.4f}, auc = {np.mean(auc):.4f}±{np.std(auc):.4f}, specificity = {np.mean(spe):.4f}±{np.std(spe):.4f}")
    # return np.mean(acc), np.mean(f1), np.mean(auc), np.mean(spe)

        
        

def set_up_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='COBRE')
    parser.add_argument('--modality', type=str, default='fmri')
    parser.add_argument('--dims', type=list, nargs='+', default=[128,128])
    parser.add_argument('--comm', type=int, default=7)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--commlr', type=float, default=1e-3)
    parser.add_argument('--commwd', type=float, default=1e-5)
    parser.add_argument('--act', type=str, default='leakyrelu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_up_seed(args.seed)
    dataset = dataloader.BrainDataset(args.root, args.dataset, args.modality)
    dataLoader = DataLoader(dataset, batch_size=16, shuffle=True)

    net = build.build_CommunityContrast_Model(dataset[0].x.shape[0], args.dims, args.tau, args.act, args.comm)
    opt = torch.optim.AdamW(net.parameters(), lr=args.commlr, weight_decay=args.commwd)

    # Train
    train_eval(net, opt, dataLoader, args.epochs)


    

    