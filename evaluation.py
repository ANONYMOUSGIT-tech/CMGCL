import torch
import dataloader

import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from torch.optim import AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(torch.nn.Module):
    def __init__(self, in_channel, num_classes=2):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, num_classes)
        )
    def forward(self, input):
      return self.net(input)   


def calculate_specificity(y_true, y_pred):
    """
    Calculate specificity given true labels and predicted labels.
    
    Args:
        y_true (list or array): Ground truth binary labels (0 or 1).
        y_pred (list or array): Predicted binary labels (0 or 1).
        
    Returns:
        float: Specificity score.
    """
    # Compute confusion matrix
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate specificity
    specificity = tn / (tn + fp)
    
    return specificity

def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    acc = []
    f1 = []
    auc = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)

        classifier.fit(x_train, y_train)
        acc.append(accuracy_score(y_test, classifier.predict(x_test)))
        f1.append(f1_score(y_test, classifier.predict(x_test)))
        auc.append(roc_auc_score(y_test, classifier.predict(x_test)))
    
    return acc, f1, auc

def mlp_evaluator(x, y, learning_rate=1e-3):
        
    results = [[],[],[],[]]
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(x, y):

        model = MLP(x.shape[1]).to(device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        data_loader = DataLoader(dataloader.BRDataset(x[train_index], y[train_index]), batch_size=32, shuffle=True)
        data_loader_eval = DataLoader(dataloader.BRDataset(x[test_index], y[test_index]), batch_size=32, shuffle=True)

        bestLoss = 1e9
        checkPoint = model.state_dict()
        for _ in range(0, 100):
            loss = train(model, optimizer, loss_function, data_loader)

            if loss < bestLoss:
                checkPoint = model.state_dict()
                bestLoss = loss

        model.load_state_dict(checkPoint)
        acc, f1ma, auc, specificity = test(model, data_loader_eval)

        results[0].append(acc)
        results[1].append(f1ma)
        results[2].append(auc)
        results[3].append(specificity)
    return results[0], results[1], results[2], results[3]

def train(model, optimizer, loss_function, data_loader):

    model.train()
    lr = 0
    for data in data_loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()
        out = model(data[0].to(device))
        loss = loss_function(out, data[1].type(torch.LongTensor).to(device)) 
        loss.backward()  
        optimizer.step()  
        lr += loss.item()
    return lr

def test(model, data_loader_eval):
    model.eval()
    preds = []
    labels = []
    for data in data_loader_eval:  
        out = model(data[0].to(device))
        pred = out.argmax(dim=1)
        preds.extend(pred.detach().cpu())
        labels.extend(data[1])
    return accuracy_score(labels,preds), \
           f1_score(labels, preds, average = 'macro'), \
           roc_auc_score(labels, preds), \
           calculate_specificity(labels, preds)