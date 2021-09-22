import pickle
def load_obj(file_name):
    with open(file_name,'rb') as f:
        return pickle.load(f)
def read_data(data_name):
    path_to_data = '../data/citation_networks_binary/'
    adj = load_obj(path_to_data + data_name+ '_adj.pkl')
    features = load_obj(path_to_data + data_name + '_features.pkl')
    labels =load_obj(path_to_data + data_name + '_labels.pkl')
    tvt_nids = load_obj(path_to_data + data_name + '_tvt_nids.pkl')
    return adj, features, labels, tvt_nids




def new_read_data(data_name):
    path_to_data = '../new_data/'
    adj = load_obj(path_to_data + data_name+ '_adj.pkl')
    features = load_obj(path_to_data + data_name + '_features.pkl')
    labels =load_obj(path_to_data + data_name + '_labels.pkl')
#     tvt_nids = load_obj(path_to_data + data_name + '_tvt_nids.pkl')
    return adj, features, labels





        