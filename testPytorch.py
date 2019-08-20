import numpy as np
import pandas as pd
import random
import os
from os.path import expanduser
import sys
import torch.nn.functional as F

# code_base = './'
# sys.path.insert(0, code_base)
# sys.path.insert(0, code_base + 'sampler')
# sys.path.insert(0, code_base + 'models')
# sys.path.insert(0, code_base + 'APE_pytorch')

from APEpytorch.apePytorch import APEmodel
from torch import optim
from torchsummary import summary

from trainPytorch import get_data, fit

from utility import data_transformation #, transform_with_keys, generate_mixed_events
#from utility import get_entity_samplers_and_noise_prob, get_entity_type_sampler_and_mappings


#########
# Data and utility preparations
#########
# data locations
data_folder = 'DemoToyPytorch/'                      # it only contains unrunnable toy data for demonstration
data_event_file = data_folder + '/events.csv'  # historical events without any label
data_test_file = data_folder + '/test.csv'     # test (future) events with additional label column


# class DataSpec(object):
#     def __init__(self, table_transformed, type2range):
#         self.num_entity_type = len(table_transformed.columns)
#         self.num_levels = [len(table_transformed[col].unique()) for col in table_transformed.columns] #the number of levels in each entity
#
#         self.num_entity = max([max(type_range) for each, type_range in type2range.items()]) - \
#                           min([min(type_range) for each, type_range in type2range.items()]) + 1
#         self.embedding_size = 3 #TODO: change. This was num_entity in the code, but not sure we want that


class DataSpec(object):
    def __init__(self, table):
        self.num_entity_type = len(table.columns)
        self.num_levels = [len(table[col].unique()) for col in table.columns] #the number of levels in each entity

        self.num_entity = max(self.num_levels) - \
                          min(self.num_levels) + 1
        self.embedding_size = 3 #TODO: change. This was num_entity in the code, but not sure we want that


class Conf(object):
    def __init__(self):
        self.max_epoch = 10
        # self.batch_size = 512
        self.batch_size = 1
        self.num_negatives = 5
        self.emb_dim = 10
        self.loss = 'skip-gram'
        #self.loss = 'max-margin'
        self.no_weight = False
        self.ignore_noise_dist = False

if __name__== "__main__":

    # # load data
    home = expanduser('~')
    table = pd.read_csv(data_event_file)
    table_test = pd.read_csv(data_test_file)
    level_to_id = [{key:idx for idx, key in enumerate(table[col].unique())} for col in table.columns]
    print(level_to_id)
    events = []
    for idx, row in table.iterrows():
        record = [level_to_id[i][row.iloc[i]] for i in range(len(table.columns))]
        events.append(record)

    events = np.asarray(events)



    # index the entities in data
    # table_transformed, id2type_and_name, type_and_name2id, type2range = data_transformation(table)
    # print(max([max(type_range) for each, type_range in type2range.items()]))
    # print(min([min(type_range) for each, type_range in type2range.items()]))
    # # table_transformed_test = transform_with_keys(table_test, table_test.columns[:-1], type_and_name2id)
    # # # drop rows in test with NaN if there are any (imputation can also be used here)
    # # table_transformed_test = table_transformed_test.dropna()
    # #
    # # # sampler preparation
    # # type2sampler, noise_prob = get_entity_samplers_and_noise_prob(table_transformed, noise_prob_cal='logkPn@10', neg_dist='unigram')
    # # type2typeid, typeid2type, entity_type_sample, type_cad_dist = get_entity_type_sampler_and_mappings(table_transformed, neg_dist='uniform')

    #TODO load data and write levels to ids

    data_spec = DataSpec(table)
    print(f"num entity {data_spec.num_entity_type}")

    conf = Conf()
    # reload(ape)
    model = APEmodel(data_spec)

    batch_size = conf.batch_size
    # events = np.asarray([[ 0, 1, 2, 4, 6, 9, 11],
    #           [ 0, 1, 3, 5, 7, 8, 10],
    #           [ 0, 1, 3, 4, 6, 9, 11],
    #           [ 0, 1, 3, 4, 6, 9, 11],
    #           [ 0, 1, 3, 4, 6, 9, 11],
    #           [ 0, 1, 3, 4, 6, 9, 11],
    #           [ 0, 1, 3, 4, 6, 9, 11],
    #           [ 0, 1, 2, 5, 7, 8, 10],
    #           [ 0, 1, 2, 5, 7, 8, 10],
    #           [ 0, 1, 2, 5, 7, 8, 10],
    #           [ 0, 1, 2, 5, 7, 8, 10],
    #           [ 0, 1, 3, 5, 7, 8, 10]])
    # print(f"events.shape {events.shape}")

    events_label = [ 1., 1., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 1., 1, 1, 0, 0, 1, 0, 1]
    print(len(events_label))
    val_idx = [2,5,6]
    train_dl, valid_dl = get_data(events, events_label, val_idx, batch_size)
    dummy_input = next(train_dl.__iter__())
    for i, tensor in enumerate(dummy_input):
        print("dummy_input[{}]:".format(i))
        print(tensor.shape)
        print(tensor.dtype)
        print("")
    #make sure the model structure matches the implementation in keras
    # print(model)
    # summary(model, events.shape)
    # summary(model, input_sizes=[(n_lvls1,),(n_lvls2,)])

    fit(1, model, F.binary_cross_entropy, optim.SGD(model.parameters(), lr=1e-2), train_dl, valid_dl)
