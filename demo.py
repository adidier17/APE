import numpy as np
import pandas as pd
import random
import os
from os.path import expanduser
import sys
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('poster')

code_base = './'
sys.path.insert(0, code_base)
sys.path.insert(0, code_base + 'sampler')
sys.path.insert(0, code_base + 'models')

from utility import data_transformation, transform_with_keys, generate_mixed_events
from utility import get_entity_samplers_and_noise_prob, get_entity_type_sampler_and_mappings

from sklearn.metrics import average_precision_score, roc_auc_score
from metrics_ranking import eval_multiple, eval_apk

import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Layer, Input, Dense, Embedding, Flatten, Merge, AveragePooling1D, Merge, Permute, merge
from keras.regularizers import WeightRegularizer, l1, l2, activity_l2, activity_l1

import ape
from ape import get_model



#########
# Data and utility preparations
#########
# data locations
data_folder = 'demo_toy/'                      # it only contains unrunnable toy data for demonstration
data_event_file = data_folder + '/events.csv'  # historical events without any label
data_test_file = data_folder + '/test.csv'     # test (future) events with additional label column


class DataSpec(object):
    def __init__(self, table_transformed, type2range):
        self.num_entity_type = len(table_transformed.columns)
        self.num_entity = max([max(type_range) for each, type_range in type2range.iteritems()]) - \
                          min([min(type_range) for each, type_range in type2range.iteritems()]) + 1

class Conf(object):
    def __init__(self):
        self.max_epoch = 10
        # self.batch_size = 512
        self.batch_size = 2
        self.num_negatives = 5
        self.emb_dim = 10
        self.loss = 'skip-gram'
        #self.loss = 'max-margin'
        self.no_weight = False
        self.ignore_noise_dist = False

if __name__== "__main__":

    # load data
    home = expanduser('~')
    table = pd.read_csv(data_event_file)
    table_test = pd.read_csv(data_test_file)

    # index the entities in data
    table_transformed, id2type_and_name, type_and_name2id, type2range = data_transformation(table)
    table_transformed_test = transform_with_keys(table_test, table_test.columns[:-1], type_and_name2id)
    # drop rows in test with NaN if there are any (imputation can also be used here)
    table_transformed_test = table_transformed_test.dropna()

    # sampler preparation
    type2sampler, noise_prob = get_entity_samplers_and_noise_prob(table_transformed, noise_prob_cal='logkPn@10', neg_dist='unigram')
    type2typeid, typeid2type, entity_type_sample, type_cad_dist = get_entity_type_sampler_and_mappings(table_transformed, neg_dist='uniform')



    data_spec = DataSpec(table_transformed, type2range)

    conf = Conf()
    reload(ape)
    model = get_model(conf, data_spec)

    abandon_uneven_batch = False
    batch_size = conf.batch_size
    num_negatives = conf.num_negatives
    events = np.array(table_transformed)
    num_iters = np.ceil(events.shape[0] / float(batch_size)).astype(int)
    for epoch in range(1, conf.max_epoch + 1):
        np.random.shuffle(events)
        cost = 0
        entity_type_assigns = entity_type_sample(num_iters)

        for it in range(num_iters):
            neg_entity_typeid = entity_type_assigns[it]
            events_batch = events[it * batch_size: (it + 1) * batch_size]
            if abandon_uneven_batch and events_batch.shape[0] != batch_size:
                continue

            events_batch_mixed, events_noise_prob, events_label = \
                generate_mixed_events(events_batch, neg_entity_typeid, num_negatives, type2sampler, typeid2type,
                                      noise_prob)

            generate_mixed_events(events_batch, neg_entity_typeid, num_negatives, type2sampler, typeid2type, noise_prob)

    #         print 'events_batch_mixed '
    #         print events_batch_mixed
    #         print events_noise_prob
    #         print events_label

            cost += model.train_on_batch([events_batch_mixed, events_noise_prob], events_label)
        print '[INFO] epoch %d, cost: %f' % (epoch, cost), 'norm', np.sqrt(np.mean(model.get_weights()[0]**2))