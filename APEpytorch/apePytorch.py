import torch.nn as nn
import torch
import numpy as np
from fastai import layers


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

class DotProdPairs(torch.nn.Module):
    """
    Not sure if this needs a backward pass too or anything else to implement the weights
    """
    def __init__(self):
        super(DotProdPairs, self).__init__()

    def forward(self, x):
        # x_t = torch.transpose(x, 2, 1) #? Not sure this works
        x_t = x.t()
        print(f"shape of x_t {x_t.shape}")
        dot_prod = torch.mm(x, x_t) #get the dot prod of all of the pairs. (Eqn 2)
        print(f"shape of dot_prod {dot_prod.shape}")
        upper_tri = np.triu(dot_prod, k=1) #filter to only 1 copy of pairs and 0 out the rest so that weights don't update. (Check with hooks to make sure)
        print(f"shape of upper_tri {upper_tri.shape}")
        print(f"upper tri {upper_tri}")
        #flatten
        #pairs = torch.flatten(upper_tri, start_dim=1)
        # pairs = upper_tri.reshape((1, upper_tri.shape[0], upper_tri.shape[1]))
        pairs = upper_tri.flatten()
        print(f"shape flattened: {pairs.shape}")
        pairs = torch.tensor(pairs)
        print(f"as tensor: {pairs.shape}")
        return pairs


class APEmodel(nn.Module):
    """

    """
    def __init__(self, data_spec):
        super(APEmodel, self).__init__()

        #Initialize Parameters TODO replace data_spec with something more compatible with fastai
        self.num_entity_type = data_spec.num_entity_type
        # self.embedding_size = data_spec.embedding_size
        self.num_pairs = int((self.num_entity_type * (self.num_entity_type - 1))/2)  #nC2 (n choose 2 combinatorial)
        self.embedding_size = data_spec.embedding_size
        self.num_levels = data_spec.num_levels
        print(self.num_levels)
        #Define all the layers
        #Layer 1: Entity Embedding Layer, embed the input, then take a permutation of the embedding
        #self.embedding = nn.Embedding(self.num_entity_type, self.embedding_size) #I think? Need more details about torch embeddings
        self.embedding = nn.ModuleList([layers.embedding(ni, self.embedding_size) for ni in self.num_levels])

        #Layer 2: Get Pairwise dot prods
        self.embedding_pairs = DotProdPairs()

        print(f"self.num_pairs is {self.num_pairs}")
        #Layer 4: Score for pairwise interactions
        self.score = nn.Linear(self.num_entity_type*self.num_entity_type, 1) #nC2 weights, summed to 1 score

    # TODO: later import this from fastai
    def emb_sz_rule(self, n_cat: int) -> int: return min(600, round(
        1.6 * n_cat ** 0.56))  # from Jeremy Howard's latest research github.com/fastai/fastai/blob/master/fastai/tabular/data.py#L14, https://forums.fast.ai/t/size-of-embedding-for-categorical-variables/42608/2

    def forward(self, input):
        """
        Define the forward pass
        :param input:
        :return:
        """
        # entity_embedding = self.embedding(input)
        print(f"input {input}, {input[:,0]}")
        for i, e in enumerate(self.embedding):
            print(f"{i}: {input[:,i]}" )
        entity_embedding = [e(input[:,i]) for i,e in enumerate(self.embedding)]
        print(f"entity embedding len (should match num categories: {len(entity_embedding)}")
        print("size of each embedding:")
        for i in range(len(entity_embedding)):
            print(entity_embedding[i].shape)
        #cast from ModuleList to tensor so that matrix operations can be performed
        entity_embedding = torch.cat(entity_embedding)
        print(f"as tensor: {entity_embedding.shape}")
        embedding_pairs = self.embedding_pairs(entity_embedding)
        # embedding_pairs = Print(embedding_pairs)
        score = self.score(embedding_pairs)
        # score = Print(score)
        return score