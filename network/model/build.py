from network.model.module import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_CommunityContrast_Model(num_features, dims, tau, activation="relu", num_comm=7, gtau=0.1):
    model = CommunityContrast(num_features, dims, tau, activation, num_comm=num_comm, gtau=gtau).to(device)
    model.apply(init_weights)
    return model

