import os.path as osp
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
from model.model_ import *
from model.TRM2_lstm_sat_am_pre import *

fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


'''
Note that training and test datasets are the same as GraphDTA
Please see: https://github.com/thinng/GraphDTA

'''

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)   
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index


VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

def seqs2int(target):

    return [VOCAB_PROTEIN[s] for s in target] 
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  


class GATADataset(InMemoryDataset):

    def __init__(self, root,Type ="train",
                 transform=None, 
                 pre_transform=None,
                 pre_filter=None,
                 max_length=512,
                 device='cuda:0'):
        super(GATADataset,self).__init__(root, transform, pre_transform, pre_filter)
       
        
        
        if Type == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        if Type == "test":
            self.data, self.slices = torch.load(self.processed_paths[1])
        if Type == "val":
            self.data, self.slices = torch.load(self.processed_paths[2])
         

        self.max_length=max_length
        self.device = device
     
      

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process_data(self,xd, xt, xt2, y,smile_graph,device='cuda:0'):#,get_mol_feat,get_prot_feat):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)

        
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smi = xd[i] # "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"  <class 'numpy.str_'>
            sequence = xt[i] # [12.  1.  1. 21.  9. 11                  <class 'numpy.ndarray'>
            prot_sequence = xt2[i]# "MRGARGAWDFLCVLLLLLRVQTGSSQPSVSP"   <class 'numpy.str_'>
            label = y[i]
      
            
            if smi in smile_graph:
                c_size, features, edge_index = smile_graph[smi]
                
                
                data = DATA.Data(x=torch.Tensor(features).to(device),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0).to(device),
                                    y=torch.FloatTensor([label]).to(device)
                                    )
                
                data.target = torch.LongTensor([sequence]).to(device)
                data.SMILES=smi
                data.FASTA= prot_sequence
                data.__setitem__('c_size', torch.LongTensor([c_size]).to(device))
                data_list.append(data)
        return data_list
                      
    def process(self):
        
        df_train = pd.read_csv(self.raw_paths[0])
        df_test = pd.read_csv(self.raw_paths[1])
        #df_val = pd.read_csv(self.raw_paths[2])
        df = pd.concat([df_train, df_test])
        smiles = df['compound_iso_smiles'].unique()
        
        train_drugs, train_prots_2,  train_Y = list(df_train['compound_iso_smiles']),list(df_train['target_sequence']),list(df_train['affinity'])
        XT = [seq_cat(t) for t in train_prots_2]
        train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
        

        test_drugs, test_prots_2,  test_Y = list(df_test['compound_iso_smiles']),list(df_test['target_sequence']),list(df_test['affinity'])
        XT = [seq_cat(t) for t in test_prots_2]
        test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
        
        smile_graph = {}
        for smile in smiles:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
            


            
        train_list=self.process_data(xd=train_drugs,
                                     xt=train_prots,
                                     xt2=train_prots_2,
                                     y=train_Y,
                                     smile_graph=smile_graph)
        test_list = self.process_data(xd=test_drugs, 
                                      xt=test_prots, 
                                      xt2=test_prots_2,
                                      y=test_Y,
                                      smile_graph=smile_graph)

        if self.pre_filter is not None:
            train_list = [train for train in train_list if self.pre_filter(train)]
            test_list = [test for test in test_list if self.pre_filter(test)]
            #val_list = [val for val in val_list if self.pre_filter(val)]

        if self.pre_transform is not None:
            train_list = [self.pre_transform(train) for train in train_list]
            test_list = [self.pre_transform(test) for test in test_list]
            #val_list = [self.pre_transform(val) for val in val_list]
        print('Graph construction done. Saving to file.')
        
        data, slices = self.collate(train_list)
        torch.save((data, slices), self.processed_paths[0])       
        data, slices = self.collate(test_list)
        torch.save((data, slices), self.processed_paths[1])
        #data, slices = self.collate(val_list)
        #torch.save((data, slices), self.processed_paths[2])        
        
    
    def get_nodes(self, g, Max_node):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            # 5 more
            h_t.append(d['ExplicitValence'])
            h_t.append(d['FormalCharge'])
            h_t.append(d['ImplicitValence'])
            h_t.append(d['NumExplicitHs'])
            h_t.append(d['NumRadicalElectrons'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def get_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]

            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t
        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def mol2graph(self, smile, Max_node=50):
        self.Max_node = Max_node
        mol = Chem.MolFromSmiles(smile)
        feats = chem_feature_factory.GetFeaturesForMol(mol)
        g = nx.DiGraph()

        # Create nodes
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),

                       # 5 more node features
                       ExplicitValence=atom_i.GetExplicitValence(),
                       FormalCharge=atom_i.GetFormalCharge(),
                       ImplicitValence=atom_i.GetImplicitValence(),
                       NumExplicitHs=atom_i.GetNumExplicitHs(),
                       NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['acceptor'] = 1

        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               # 1 more edge features 2 dim
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )

        node_attr = self.get_nodes(g,self.Max_node)
        edge_index, edge_attr = self.get_edges(g)

        return node_attr, edge_index, edge_attr


if __name__ == "__main__":
    #GATADataset('data_raw/davis')
    #GATADataset('data_raw/kiba')
    #GATADataset('/home/star/autodl-tmp/GraphDTA-master2/GtatDTA/data_raw/test')
    #GATADataset('/home/star/autodl-tmp/GraphDTA-master2/GtatDTA/data_raw/davis')
    GATADataset('data_raw/kiba')
