from textattack.attacks import Attack, AttackResult
import numpy as np
import math

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Current_node: A Node Type
# gRAVE score: A list with size N
# Current_feature: Numpy bool array? A set type?
# Params: Dictionary

# gRAVE[0] -> Count 
# gRAVE[1] -> Score

# Tree: A dictionary of nodes
class Node:
    def __init__(self,feature_set):
        self.feature_set = feature_set
        self.f_size = self.feature_set.sum()
        self.childrens = {}
        self.T_f = .0
        self.av = .0
        self.allowed_features = feature_set.nonzero()
        self.lrave_count = np.array(feature_set.shape)
        self.lrave_reward = np.array(feature_set.shape).astype(float)
        self.lrave_variance = np.array(feature_set.shape).astype(float)
        self.lrave_score = np.array(feature_set.shape).astype(float)

class Tree:
    def __init__(self,nsize):
        self.tree = {}
        root = np.array([False] * nsize,dtype=bool)
        # self.tree[str(root)] = Node(root)
        self.tree[root.tobytes()] = Node(root)
    def find(self,feature_set):
        if feature_set.tobytes() in self.tree:
            return self.tree[feature_set.tobytes()]
        else:
            return None
    def save(self,feature_set,node):
        self.tree[feature_set.tobytes()] = node

class MCTS(Attack):
    def __init__(self, model, rewardfunc, maxdepth, lcount, nsize):
        super().__init__(model)
        self.bst = 0
        self.params = dotdict({})
        self.params.get_reward = rewardfunc
        self.params.b = 0.26
        self.params.cl = 0.5
        self.params.ce = 0.5
        self.params.max_depth = maxdepth
        self.params.nrand = 50
        self.lcount = lcount
        self.gRAVE = (np.array([.0]* nsize),np.array([.0]*nsize))
        self.tree = Tree(nsize)
        self.nsize = nsize

    def update_Node(self, node, fi, current_features, reward_V):
        node.av = (node.av * node.T_f + reward_V)/(node.T_f+1)
        node.T_f += 1
        lrave_score_pre = node.lrave_score[fi]
        node.lrave_score[fi] = (node.lrave_score[fi] * node.lrave_count[fi] + reward_V) / (node.lrave_count[fi] + 1)
        node.lrave_variance[fi] = math.sqrt( ((reward_V - node.lrave_score[fi]) * (reward_V - lrave_score_pre) + node.lrave_count[fi] * node.lrave_variance[fi] * node.lrave_variance[fi])/(node.lrave_count[fi]+1))
        node.lrave_count[fi] = node.lrave_count[fi] + 1

    def update_gRAVE(self, F, reward_V):
        # update gRAVE score for each feature of feature subset F, by adding the reward_V the the score
        for fi in range(self.gRAVE[0].shape[0]):
            if F[fi]:
                self.gRAVE[0][fi] = (self.gRAVE[0][fi]*self.gRAVE[1][fi] + reward_V)/(self.gRAVE[1][fi] + 1)
                self.gRAVE[1][fi] += 1 


    def UCB(self, node):
    # b<1
        d = len(node.allowed_features)  # feature subset size
        f = node.feature_set.shape[0] # number of features
        nrand = self.params.nrand

        if (node.T_f < nrand): # we perform random exploration fort the first 50 visits
            return -1 #//-1 indicate that we don't to want chose any feature

        if (pow((node.T_f+1), self.params.b)-pow(node.T_f, self.params.b)>1):
            if not node.allowed_features:
                return -1
            else:
                for ft in range(f):
                    if not ft in allowed_features:
                        beta = self.params.cl/(self.params.cl + node.lrave_count[ft])
                        rst = (1-beta) * self.lrave_score[ft] + beta * self.gRAVE[0][ft]
                        if rst>nowbest:
                            ft_now = ft
                            nowbest = rst
                node.allowed_features.append(ft)
        else:
            return -1        

        UCB_max_score = 0
        UCB_max_feature = 0
        for next_node in node.allowed_features: # computing UCB for each feature
            UCB_Score = node.mu_f[next_node] + math.sqrt( params.ce*log(node.T_F)/node.t_f[next_node]  *  min(0.25 ,  pow(node.sg_f[next_node],2) + math.sqrt(2*math.log(node.T_F)/node.t_f[fi]) ))
            if UCB_Score>UCB_max_feature:
                UCB_max_feature = UCB_Score
                UCB_max_feature = next_node
        return UCB_max_feature


    def update_Tree_And_Get_Address(self, current_features):
        if not self.tree.find(current_features):
            node = Node(current_features)
            self.tree.save(current_features, node)
        else:
            node = self.tree.find(current_features)
        return node

    def iterate(self, current_features, depth):
        if depth>=self.params.max_depth:
            reward_V = self.params.get_reward(current_features)
            self.update_gRAVE(current_features, reward_V)
        else:
            next_node = self.update_Tree_And_Get_Address(current_features)

            if (next_node.T_f != 0):
                fi = self.UCB(next_node)
                if (fi==-1): # it means that no feature has been selected and that we are going to perform random exploration
                    depth = current_features.sum()
                    reward_V = self.iterate_random(self.tree, current_features)
                    self.update_gRAVE(current_features, reward_V)
                else: #add the feature to the feature set
                    current_features[fi] = True
                    reward_V = self.iterate(current_features, depth+1)
            else:
                depth_now = current_features.sum()
                reward_V = self.iterate_random(self.tree, current_features)
                self.update_gRAVE(current_features, reward_V)
                fi = -1 # indicate that random exploration has been perform and thus no feature selected
            self.update_Node(next_node, fi, current_features, reward_V)
        return reward_V

    def iterate_random(self, tree, current_features):
        f_num = current_features.shape[0]
        f_size  = current_features.sum()
        while (f_size < self.params.max_depth):
            if (f_num<=f_size):
                break
            #chose a random feature that is not already in the feature subset, and put its value to one (and not the stopping feature)
            t = 0
            it =  int(np.random.rand() * (f_num-f_size))
            for i in range(f_num):
                if not current_features[i] and t==it:
                    it = i
                    break
                elif not current_features[i]:
                    t = t + 1
            current_features[it] = True
            f_size += 1

        return self.params.get_reward(self, current_features)

    def reward(self, f):
        #global bst
        score = 0
        for i in range(f.shape[0]):
            if f[i]:
                if i % 3 ==0:
                    score = score + 10
                elif i % 3==1:
                    score = score - 5
        for i in range(f.shape[0]):
            if f[i]:
                if i % 5 ==0:
                    score = score + 10
                elif i % 5==1:
                    score = score - 5
        for i in range(f.shape[0]-1):
            if f[i] and f[i+1]:
                score = score + 20
        if score > self.bst:
            print(f, self.bst)
            self.bst = score
        return score

    def runmcts(self):
        for i in range(self.lcount):
            current_features = np.array([False] * self.nsize, dtype = bool)
            self.iterate(current_features, 0)  

    
if __name__  == '__main__':
    m_attk = MCTS(None, MCTS.reward, 10, 100000, 100) #params: model, rewardfunc, maxdepth, lcount, nsize
    m_attk.runmcts()
    print(m_attk.gRAVE)