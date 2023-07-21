import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm

from utils import *
from model import *
from data import *
from args import Args
import create_graphs
from train import *


#######################Function present in train.py###########################
# def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
#     rnn.hidden = rnn.init_hidden(test_batch_size)
#     rnn.eval()
#     output.eval()

#     # generate graphs
#     max_num_node = int(args.max_num_node)
#     y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
#     x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
#     for i in range(max_num_node):
#         h = rnn(x_step)
#         # output.hidden = h.permute(1,0,2)
#         hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).cuda()
#         output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
#                                   dim=0)  # num_layers, batch_size, hidden_size
#         x_step = Variable(torch.zeros(test_batch_size,1,args.max_prev_node)).cuda()
#         output_x_step = Variable(torch.ones(test_batch_size,1,1)).cuda()
#         for j in range(min(args.max_prev_node,i+1)):
#             output_y_pred_step = output(output_x_step)
#             output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
#             x_step[:,:,j:j+1] = output_x_step
#             output.hidden = Variable(output.hidden.data).cuda()
#         y_pred_long[:, i:i + 1, :] = x_step
#         rnn.hidden = Variable(rnn.hidden.data).cuda()
#     y_pred_long_data = y_pred_long.data.long()

#     # save graphs as pickle
#     G_pred_list = []
#     for i in range(test_batch_size):
#         adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#         G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#         G_pred_list.append(G_pred)

#     return G_pred_list

args = Args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
graphs = create_graphs.create(args)
args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])

rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output).cuda()
output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True, output_size=1).cuda()

test_rnn_epoch(epoch=None, args=args, rnn=rnn, output=output, test_batch_size=16)
print('Done')