r"""
This file contains preset partitioners for the benchmarks.
All the Partitioner should implement the method `__call__(self, data)`
where `data` is the dataset to be partitioned and the return is a list of the partitioned result.

For example, The IIDPartitioner.__call__ receives a indexable object (i.e. instance of torchvision.datasets.mnsit.MNSIT)
and I.I.D. selects samples' indices in the original dataset as each client's local data.
The list of list of sample indices are finally returnerd (e.g. [[0,1,2,...,1008], ...,[25,23,98,...,997]]).

To use the partitioner, you can specify Partitioner in the configuration dict for `flgo.gen_task`.
 Example 1: passing the parameter of __init__ of the Partitioner through the dict `para`
>>>import flgo
>>>config = {'benchmark':{'name':'flgo.benchmark.mnist_classification'},
...            'partitioner':{'name':'IIDPartitioner', 'para':{'num_clients':20, 'alpha':1.0}}}
>>>flgo.gen_task(config, './test_partition')
"""
import warnings
from abc import abstractmethod, ABCMeta
import random

import networkx as nx
import numpy as np
import collections
import torch
from torch.utils.data import ConcatDataset
try:
    import community.community_louvain
except:
    pass

class AbstractPartitioner(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class BasicPartitioner(AbstractPartitioner):
    """This is the basic class of data partitioner. The partitioner will be directly called by the
    task generator of different benchmarks. By overwriting __call__ method, different partitioners
    can be realized. The input of __call__ is usually a dataset.
    """
    def __call__(self, *args, **kwargs):
        return

    def register_generator(self, generator):
        r"""Register the generator as an self's attribute"""
        self.generator = generator

    def data_imbalance_generator(self, num_clients, datasize, imbalance=0, minvol=1):
        r"""
        Split the data size into several parts

        Args:
            num_clients (int): the number of clients
            datasize (int): the total data size
            imbalance (float): the degree of data imbalance across clients
            minvol (int): the minimal size of dataset
        Returns:
            a list of integer numbers that represents local data sizes
        """
        if imbalance == 0:
            samples_per_client = [int(datasize / num_clients) for _ in range(num_clients)]
            for _ in range(datasize % num_clients): samples_per_client[_] += 1
        else:
            imbalance = max(0.1, imbalance)
            sigma = imbalance
            mean_datasize = datasize / num_clients
            mu = np.log(mean_datasize) - sigma ** 2 / 2.0
            samples_per_client = np.random.lognormal(mu, sigma, (num_clients)).astype(int)
            crt_data_size = sum(samples_per_client)
            total_delta = np.abs(crt_data_size-datasize)
            thresold = max(int(total_delta/10), 1)
            delta = max(min(int(0.1 * thresold), 10), 1)
            # force current data size to match the total data size
            while crt_data_size != datasize:
                if crt_data_size - datasize >= thresold:
                    maxid = np.argmax(samples_per_client)
                    maxvol = samples_per_client[maxid]
                    new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    while min(new_samples) > maxvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[maxid] + s - datasize) for s in new_samples])
                    samples_per_client[maxid] = new_samples[new_size_id]
                elif crt_data_size - datasize >= delta:
                    maxid = np.argmax(samples_per_client)
                    if samples_per_client[maxid]>=delta:
                        samples_per_client[maxid] -= delta
                    elif samples_per_client[maxid]>1:
                        samples_per_client[maxid] -= 1
                elif crt_data_size - datasize > 0:
                    maxid = np.argmax(samples_per_client)
                    crt_delta = (crt_data_size - datasize)
                    if samples_per_client[maxid]>=crt_delta:
                        samples_per_client[maxid] -= crt_delta
                    elif samples_per_client[maxid]>=minvol:
                        samples_per_client[maxid] -= (crt_delta-minvol)
                    else:
                        warnings.warn("Failed to keep the minvol of clients' training data to be larger than {}".format(minvol))
                        if samples_per_client[maxid] > 1:
                            samples_per_client[maxid] -=1
                        else:
                            raise RuntimeError("Failed to generate distribution due to the conflicts of imbalance and num_clients. Please try to decrease the imbalance term or decrease the number of clients. ")
                elif datasize - crt_data_size >= thresold:
                    minid = np.argmin(samples_per_client)
                    minvol = samples_per_client[minid]
                    new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    while max(new_samples) < minvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[minid] + s - datasize) for s in new_samples])
                    samples_per_client[minid] = new_samples[new_size_id]
                elif datasize - crt_data_size >= delta:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += delta
                else:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += (datasize - crt_data_size)
                crt_data_size = sum(samples_per_client)
            # let the minimal data size to be larger than 0
            while min(samples_per_client)==0:
                zero_client_idx = np.argmin(samples_per_client)
                maxid = np.argmax(samples_per_client)
                samples_per_client[maxid] -=1
                samples_per_client[zero_client_idx] += 1
            assert datasize==sum(samples_per_client) and min(samples_per_client)>0
        return samples_per_client

class IIDPartitioner(BasicPartitioner):
    """`Partition the indices of samples in the original dataset indentically and independently.

    Args:
        num_clients (int, optional): the number of clients
        imbalance (float, optional): the degree of imbalance of the amounts of different local data (0<=imbalance<=1)
    """
    def __init__(self, num_clients=100, imbalance=0):
        self.num_clients = num_clients
        self.imbalance = imbalance

    def __str__(self):
        name = "iid"
        if self.imbalance > 0: name += '_imb{:.1f}'.format(self.imbalance)
        return name

    def __call__(self, data):
        samples_per_client = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance)
        d_idxs = np.random.permutation(len(data))
        local_datas = np.split(d_idxs, np.cumsum(samples_per_client))[:-1]
        local_datas = [di.tolist() for di in local_datas]
        return local_datas

class DirichletPartitioner(BasicPartitioner):
    """`Partition the indices of samples in the original dataset according to Dirichlet distribution of the
    particular attribute. This way of partition is widely used by existing works in federated learning.

    Args:
        num_clients (int, optional): the number of clients
        alpha (float, optional): `alpha`(i.e. alpha>=0) in Dir(alpha*p) where p is the global distribution. The smaller alpha is, the higher heterogeneity the data is.
        imbalance (float, optional): the degree of imbalance of the amounts of different local data (0<=imbalance<=1)
        error_bar (float, optional): the allowed error when the generated distribution mismatches the distirbution that is actually wanted, since there may be no solution for particular imbalance and alpha.
        index_func (func, optional): to index the distribution-dependent (i.e. label) attribute in each sample.
    """
    def __init__(self, num_clients=100, alpha=1.0, error_bar=1e-6, imbalance=0, index_func=lambda X:[xi[-1] for xi in X], minvol=1):
        self.num_clients = num_clients
        self.alpha = alpha
        self.imbalance = imbalance
        self.index_func = index_func
        self.minvol = minvol
        self.error_bar = error_bar

    def __str__(self):
        name = "dir{:.2f}_err{}".format(self.alpha, self.error_bar)
        if self.imbalance > 0: name += '_imb{:.1f}'.format(self.imbalance)
        return name

    def __call__(self, data):
        attrs = self.index_func(data)
        num_attrs = len(set(attrs))
        samples_per_client = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance, minvol=self.minvol)
        # count the label distribution
        lb_counter = collections.Counter(attrs)
        lb_names = list(lb_counter.keys())
        p = np.array([1.0 * v / len(data) for v in lb_counter.values()])
        lb_dict = {}
        attrs = np.array(attrs)
        for lb in lb_names:
            lb_dict[lb] = np.where(attrs == lb)[0]
        proportions = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
        while np.any(np.isnan(proportions)):
            proportions = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
        sorted_cid_map = {k: i for k, i in zip(np.argsort(samples_per_client), [_ for _ in range(self.num_clients)])}
        error_increase_interval = 500
        max_error = self.error_bar
        loop_count = 0
        crt_id = 0
        crt_error = 100000
        while True:
            if loop_count >= error_increase_interval:
                loop_count = 0
                max_error = max_error * 10
            # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
            mean_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
            mean_prop = mean_prop / mean_prop.sum()
            error_norm = ((mean_prop - p) ** 2).sum()
            if crt_error - error_norm >= max_error:
                print("Approximation Error: {:.8f}".format(error_norm))
                crt_error = error_norm
            if error_norm <= max_error:
                break
            excid = sorted_cid_map[crt_id]
            crt_id = (crt_id + 1) % self.num_clients
            sup_prop = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
            del_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
            del_prop -= samples_per_client[excid] * proportions[excid]
            for i in range(error_increase_interval - loop_count):
                alter_norms = []
                for cid in range(self.num_clients):
                    if np.any(np.isnan(sup_prop[cid])):
                        continue
                    alter_prop = del_prop + samples_per_client[excid] * sup_prop[cid]
                    alter_prop = alter_prop / alter_prop.sum()
                    error_alter = ((alter_prop - p) ** 2).sum()
                    alter_norms.append(error_alter)
                if min(alter_norms) < error_norm:
                    break
            if len(alter_norms) > 0 and min(alter_norms) < error_norm:
                alcid = np.argmin(alter_norms)
                proportions[excid] = sup_prop[alcid]
            loop_count += 1
        local_datas = [[] for _ in range(self.num_clients)]
        self.dirichlet_dist = []  # for efficiently visualizing
        for lb in lb_names:
            lb_idxs = lb_dict[lb]
            lb_proportion = np.array([pi[lb_names.index(lb)] * si for pi, si in zip(proportions, samples_per_client)])
            lb_proportion = lb_proportion / lb_proportion.sum()
            lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
            lb_datas = np.split(lb_idxs, lb_proportion)
            self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
            local_datas = [local_data + lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]
        self.dirichlet_dist = np.array(self.dirichlet_dist).T
        for i in range(self.num_clients): np.random.shuffle(local_datas[i])
        len_dist = [len(d) for d in local_datas]
        while min(len_dist)<=self.minvol:
            min_did = np.argmin(len_dist)
            max_did = np.argmax(len_dist)
            max_d = local_datas[max_did]
            min_d = local_datas[min_did]
            if len(max_d)<=self.minvol:
                raise RuntimeError("The number of clients is too large to distribute enough samples to each client when minvol=={}. Please decrease the number of clients".format(self.minvol))
            min_d.extend(max_d[:1])
            max_d = max_d[1:]
            local_datas[min_did] = min_d
            local_datas[max_did] = max_d
            len_dist = [len(d) for d in local_datas]
        self.local_datas = local_datas
        return local_datas

class DiversityPartitioner(BasicPartitioner):
    """`Partition the indices of samples in the original dataset according to numbers of types of a particular
    attribute (e.g. label) . This way of partition is widely used by existing works in federated learning.

    Args:
        num_clients (int, optional): the number of clients
        diversity (float, optional): the ratio of locally owned types of the attributes (i.e. the actual number=diversity * total_num_of_types)
        imbalance (float, optional): the degree of imbalance of the amounts of different local data (0<=imbalance<=1)
        index_func (int, optional): the index of the distribution-dependent (i.e. label) attribute in each sample.
    """
    def __init__(self, num_clients=100, diversity=1.0, index_func=lambda X:[xi[-1] for xi in X]):
        self.num_clients = num_clients
        self.diversity = diversity
        self.index_func = index_func

    def __str__(self):
        name = "div{:.1f}".format(self.diversity)
        return name

    def __call__(self, data):
        labels = self.index_func(data)
        num_classes = len(set(labels))
        dpairs = [[did, lb] for did, lb in zip(list(range(len(data))), labels)]
        num = max(int(self.diversity * num_classes), 1)
        K = num_classes
        local_datas = [[] for _ in range(self.num_clients)]
        if num == K:
            for k in range(K):
                idx_k = [p[0] for p in dpairs if p[1] == k]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, self.num_clients)
                for cid in range(self.num_clients):
                    local_datas[cid].extend(split[cid].tolist())
        else:
            times = [0 for _ in range(num_classes)]
            contain = []
            for i in range(self.num_clients):
                current = []
                j = 0
                while (j < num):
                    mintime = np.min(times)
                    ind = np.random.choice(np.where(times == mintime)[0])
                    if (ind not in current):
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            for k in range(K):
                idx_k = [p[0] for p in dpairs if p[1] == k]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[k])
                ids = 0
                for cid in range(self.num_clients):
                    if k in contain[cid]:
                        local_datas[cid].extend(split[ids].tolist())
                        ids += 1
        return local_datas

