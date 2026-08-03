import numpy as np
import torch
from grid2op.Environment import Environment
from grid2op.Action import ActionSpace
from sknetwork.clustering import Louvain
from scipy.sparse import csr_matrix


class ActionConverter:
    def __init__(self, env:Environment) -> None:
        self.action_space = env.action_space
        self.env = env
        self.sub_mask = []
        self.init_sub_topo()
        self.init_action_converter()

    def init_sub_topo(self):
        self.subs = np.flatnonzero(self.action_space.sub_info)
        self.sub_to_topo_begin, self.sub_to_topo_end = [], [] # These lists will eventually store the starting and ending indices, respectively, for each actionable substation's topology data within the environment's overall topology information.
        idx = 0 # This variable will be used to keep track of the current position within the overall topology data
        
        for num_topo in self.action_space.sub_info: # The code can efficiently extract the relevant portion of the overall topology data that specifically applies to the given substation
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)

    def init_action_converter(self):
        self.actions = [self.env.action_space({})]
        self.n_sub_actions = np.zeros(len(self.action_space.sub_info), dtype=int)
        for i, sub in enumerate(self.subs):
            
            # Generating Topology Actions
            topo_actions = self.action_space.get_all_unitary_topologies_set(self.action_space, sub) # retrieves all possible topology actions for the current substation using the get_all_unitary_topologies_set method of the action_space object
            self.actions += topo_actions  # Appends the topology actions for the current substation to the actions list.
            self.n_sub_actions[i] = len(topo_actions) # Stores the number of topology actions for the current substation in the n_sub_actions array
            self.sub_mask.extend(range(self.sub_to_topo_begin[sub], self.sub_to_topo_end[sub])) # Extends the sub_mask list with indices corresponding to the topologies of the current substation.
        
        self.sub_pos = self.n_sub_actions.cumsum() 
        self.n = sum(self.n_sub_actions)

    def act(self, action:int):
        return self.actions[action]
    
    def action_idx(self, action):
        return self.actions.index(action)
    
    def one_hot_encode(tensor, num_classes):
        """
        One-hot encode a tensor of indices.
        
        Args:
            tensor (torch.Tensor): Tensor containing class indices (e.g., tensor([104], device='cuda:0')).
            num_classes (int): Total number of classes.
            
        Returns:
            torch.Tensor: One-hot encoded tensor.
        """
        # Ensure tensor is long type for indexing
        tensor = tensor.long()
        
        # Create a one-hot encoded tensor
        one_hot = torch.zeros(tensor.size(0), num_classes, device=tensor.device)
        one_hot.scatter_(1, tensor.unsqueeze(1), 1)
        
        return one_hot





class MADiscActionConverter:
    def __init__(self, env, sub_list) -> None:
        self.action_space = env.action_space
        self.env = env
        self.sub_mask = []
        self.sub_list = sub_list
        self.init_cluster_action_converter()

    def init_cluster_action_converter(self):
        """
        Initialize cluster action converters based on the number of clusters.
        
        Parameters:
        env: The environment object which contains the action space.
        action_domains (dict): A dictionary where keys are agent names and values are lists of substation IDs.
        
        Returns:
        list: A list of lists, where each inner list contains actions for a specific cluster.
        """
        self.cluster_actions = []

        cluster_action_list = []
        for sub in self.sub_list:
            sub_actions = self.env.action_space.get_all_unitary_topologies_set(self.env.action_space, sub_id=sub)
            cluster_action_list.extend(sub_actions)
        self.cluster_actions.append(cluster_action_list)


    def act(self, action:int):
        return self.cluster_actions[0][action]
    
    def action_idx(self, action):
        return self.cluster_actions[0].index(action)
    
    def action_size(self):
        return len(self.cluster_actions[0])
    