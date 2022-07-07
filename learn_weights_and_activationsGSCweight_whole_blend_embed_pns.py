import json
import numpy as np
import torch
import torch.nn as nn
import time
import math
import collections

use_frequency = True

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#with open('shuffled_class_dict.json', 'r') as f1:
with open('culled_class_dict_bitcost_compar.json', 'r') as f1:
    class_dict = json.load(f1)

for k in list(class_dict):
    num_pns = len(class_dict[k][1])
    break

exps = sorted(set(e for k in list(class_dict) for e in class_dict[k][1]))
print('e', exps)

torch.set_printoptions(sci_mode=False)

num_exps = len(exps)
print('There are', num_pns, 'pns')
print('There are', num_exps, 'exps')
exponent2idx = {exp: i for i, exp in enumerate(exps)}

stem_embedding_dim = 1
num_epochs = 400
verbose = False
lexeme_index_to_identifier = collections.defaultdict(lambda: None)
targets = np.zeros((len(class_dict), num_pns))
for i, k in enumerate(list(class_dict)):
    pattern_set = class_dict[k][1]
    lexeme_index_to_identifier[str(i)] = k
    for j in range(num_pns):
        targets[i, j] = exponent2idx[pattern_set[j]]

class LearnActivationsWeights(nn.Module):

    def __init__(self, num_exps, num_pns, class_dict, stem_embedding_dim):
        super(LearnActivationsWeights, self).__init__()
        self.class_dict = class_dict
        self.embeddings_stems = nn.Embedding(len(self.class_dict), num_exps)
        self.embeddings_pn = nn.Embedding(num_pns, num_exps)
        self.dep_bias = nn.Embedding(1, num_exps)
        self.sigmoid = nn.Sigmoid()
        self.dep = torch.nn.Parameter(torch.rand((1), requires_grad=True, dtype=torch.float))
        self.n_digits_stem = 3
        self.n_digits_pn = 3

    def forward(self, stem_index, round_for_testing=False):
        self.embs_stem = self.sigmoid(self.embeddings_stems(stem_index))
        self.embs_stem = torch.unsqueeze(self.embs_stem.repeat(num_pns),0)# / 2
        self.embs_pn_0  = self.sigmoid(self.embeddings_pn(torch.tensor(0))) #/ 2
        self.embs_pn_1  = self.sigmoid(self.embeddings_pn(torch.tensor(1))) #/ 2
        self.embs_pn_2  = self.sigmoid(self.embeddings_pn(torch.tensor(2))) #/ 2
        self.embs_pn_3  = self.sigmoid(self.embeddings_pn(torch.tensor(3))) #/ 2
        self.embs_pn_4  = self.sigmoid(self.embeddings_pn(torch.tensor(4))) #/ 2
        self.embs_pn = torch.unsqueeze(torch.cat((self.embs_pn_0, self.embs_pn_1, self.embs_pn_2, self.embs_pn_3, self.embs_pn_4), 0),0)
        coalesced = (self.embs_stem + self.embs_pn)
        self.bias = self.dep_bias(torch.tensor(0)).repeat(num_pns)


        max_reward = coalesced
        dep_penalty = (torch.ones(coalesced.size()) - coalesced)

        if not round_for_testing:
            return max_reward - dep_penalty, max_reward, dep_penalty, coalesced
        else:
            with torch.no_grad():
                self.embs_stem = torch.round(self.sigmoid(self.embeddings_stems(stem_index))*2**self.n_digits_stem)/2**self.n_digits_stem
                self.embs_stem = torch.unsqueeze(self.embs_stem.repeat(num_pns),0)# / 2
                self.embs_pn_0  = torch.round(self.sigmoid(self.embeddings_pn(torch.tensor(0)))*2**self.n_digits_pn)/2**self.n_digits_pn #/ 2
                self.embs_pn_1  = torch.round(self.sigmoid(self.embeddings_pn(torch.tensor(1)))*2**self.n_digits_pn)/2**self.n_digits_pn   #/ 2
                self.embs_pn_2  = torch.round(self.sigmoid(self.embeddings_pn(torch.tensor(2)))*2**self.n_digits_pn)/2**self.n_digits_pn   #/ 2
                self.embs_pn_3  = torch.round(self.sigmoid(self.embeddings_pn(torch.tensor(3)))*2**self.n_digits_pn)/2**self.n_digits_pn   #/ 2
                self.embs_pn_4  = torch.round(self.sigmoid(self.embeddings_pn(torch.tensor(4)))*2**self.n_digits_pn)/2**self.n_digits_pn   #/ 2
                self.embs_pn = torch.unsqueeze(torch.cat((self.embs_pn_0, self.embs_pn_1, self.embs_pn_2, self.embs_pn_3, self.embs_pn_4), 0),0)
                coalesced = (self.embs_stem + self.embs_pn)
                max_reward = coalesced
                dep_penalty = (torch.ones(coalesced.size()) - coalesced)
                return max_reward - dep_penalty, max_reward, dep_penalty, coalesced


def train(class_dict, targets, net, lr, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    num_examples = len(class_dict)
    for epoch in range(num_epochs):
        ep_loss = 0.
        start_time = time.time()
        num_tested = 0
        num_correct = 0
        margins = []
        for i in torch.randperm(len(class_dict)):
            #    continue
            key = list(class_dict)[i]
            preds, max_reward, dep_penalty, coalesced = net(torch.tensor(i).clone().detach())
            loss = 0
            for j in range(num_pns):
                if use_frequency:
                    num_tested += class_dict[key][0]
                else:
                    num_tested += 1
                target = torch.tensor(targets[i,j])
                target = target.contiguous().view(-1)
                target = target.long()
                target_name = exps[int(targets[i,j])]
                pred = preds[:,j*num_exps:j*num_exps+num_exps]
                pred_name = exps[int(torch.argmax(pred).item())]
                if pred_name == target_name:
                    if use_frequency:
                        num_correct += class_dict[key][0]
                    else:
                        num_correct += 1
                    preds_ordered = list(sorted(pred.detach().numpy().tolist()[0], reverse=True))
                    margin = round(preds_ordered[0] - preds_ordered[1], 4)
                    margins.append(margin)
                else:
                    if epoch == num_epochs - 1 and verbose == True:
                        print('error', 'target is', target, 'j=', j, lexeme_index_to_identifier[str(i.detach().numpy())],  i, end=' ')
                        for val in pred.contiguous().view(-1).detach().numpy().tolist():
                            print(round(val, 3), end=' ')
                        print()
                if epoch == num_epochs - 1 and verbose == True:
                    print('Harmonies for lexeme', lexeme_index_to_identifier[str(i.detach().numpy())], end=' ')
                    for val in pred.contiguous().view(-1).detach().numpy().tolist():
                        print(round(val, 3), end=' ')
                    print()
                if use_frequency:
                    loss += criterion(pred, target) * class_dict[key][0] / 100
                else:
                    loss += criterion(pred, target)

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            ep_loss += loss.detach()

        print(epoch, ep_loss, round(time.time()-start_time, 3))
        print('Accuracy', round(num_correct / num_tested, 4), num_tested, num_correct)
        print('Margins', list(sorted(margins))[0])
        if epoch %10 ==0:
            print()
            print()
            print('Embeddings')
            for i in range(len(class_dict)):
                print(lexeme_index_to_identifier[str(i)], end=' ')
                row = net.sigmoid(net.embeddings_stems.weight[i,:])
                row_sum = torch.sum(row).detach().numpy()
                for element in row.detach().numpy().tolist():
                    if round(element, 3) == 0.0:
                        print('_', end=' ')
                    else:
                        print(round(element, 3), end=' ')
                print()
            print()
 
            
            print('Embeddings pn')
            for i in range(num_pns):
                row = net.sigmoid(net.embeddings_pn.weight[i,:])
                row_sum = torch.sum(row).detach().numpy()
                for element in row.detach().numpy():
                    if round(element, 3) == 0.0:
                        print('_', end=' ')
                    else:
                        print(round(element, 3), end=' ')
                print()
            print()

    print('Now testing')
    num_examples = len(class_dict)
    num_tested = 0
    num_correct = 0
    margins = []
    lexemes_to_exclude = []
    with torch.no_grad():
        for i in torch.randperm(len(class_dict)):
            key = list(class_dict)[i]
            preds, max_reward, dep_penalty, coalesced = net(torch.tensor(i).clone().detach(), True)
            for j in range(num_pns):
                if use_frequency:
                    num_tested += class_dict[key][0]
                else:
                    num_tested += 1
                target = torch.tensor(targets[i,j])
                target = target.contiguous().view(-1)
                target = target.long()
                target_name = exps[int(targets[i,j])]
                pred = preds[:,j*num_exps:j*num_exps+num_exps]
                pred_name = exps[int(torch.argmax(pred).item())]
                if pred_name == target_name:
                    if use_frequency:
                        num_correct += class_dict[key][0]
                    else:
                        num_correct += 1
                    preds_ordered = list(sorted(pred.detach().numpy().tolist()[0], reverse=True))
                    margin = round(preds_ordered[0] - preds_ordered[1], 4)
                    margins.append(margin)
                else:
                    #print('On testing, missed lexeme', i, 'for pn', j, 'with frequency', class_dict[key][0])
                    if class_dict[key][0] in ['1', 1] and i not in lexemes_to_exclude:
                        lexemes_to_exclude.append(i.item())
                    if epoch == num_epochs - 1 and verbose == True:
                        print('error', 'target is', target, 'j=', j, lexeme_index_to_identifier[str(i.detach().numpy())],  i, end=' ')
                        for val in pred.contiguous().view(-1).detach().numpy().tolist():
                            print(round(val, 3), end=' ')
                        print()
                if epoch == num_epochs - 1 and verbose == True:
                    print('Harmonies for lexeme', lexeme_index_to_identifier[str(i.detach().numpy())], end=' ')
                    for val in pred.contiguous().view(-1).detach().numpy().tolist():
                        print(round(val, 3), end=' ')
                    print()
        print('Accuracy', round(num_correct / num_tested, 4), num_tested, num_correct)
        print()
        print()
        print('Embeddings')
        for i in range(len(class_dict)):
            print(lexeme_index_to_identifier[str(i)], end=' ')
            row = torch.round(net.sigmoid(net.embeddings_stems(torch.tensor(i)))*2**net.n_digits_stem)/2**net.n_digits_stem
            for element in row.detach().numpy().tolist():
                if element == 0.0:
                    print('_', end=' ')
                else:
                    print(round(element, net.n_digits_stem), end=' ')
            print()
        print()
        
        print('Embeddings pn')
        for i in range(num_pns):
            row = torch.round(net.sigmoid(net.embeddings_pn(torch.tensor(i)))*2**net.n_digits_pn)/2**net.n_digits_pn
            for element in row.detach().numpy():
                if round(element, 3) == 0.0:
                    print('_', end=' ')
                else:
                    print(round(element, net.n_digits_pn), end=' ')
            print()


net = LearnActivationsWeights(num_exps, num_pns, class_dict, stem_embedding_dim)
lr = 0.003
train(class_dict, targets, net, lr, num_epochs)
