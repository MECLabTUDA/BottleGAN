import glob
import argparse
import numpy as np
import yaml

def generate(fold_folder, id, scenario='lac', n_clients=2, labeled_share=1.0, n_labeled_windows=8, n_min_windows=4,
                     n_files=1, server_n_files=4, length=2000, server_length=2000, bcss=True):

    if scenario ==  'lab':
        n_labeled_windows *= (2/3)
        n_labeled_windows = int(n_labeled_windows)

    # Get client ids and determine unlabeled ones
    client_ids = np.random.permutation(n_clients)
    labeled = np.ones(client_ids.shape)
    labeled[int(labeled_share * n_clients):] = 0
    n_labeled = int(np.sum(labeled))

    avg_windows = n_labeled_windows // n_labeled // n_files
    min_max = avg_windows - n_min_windows
    assert n_labeled_windows % n_labeled  == 0
    print(avg_windows, min_max)
    if scenario == 'lac':
        cuts = np.zeros(n_labeled, dtype=int)
        for c_id in range(0, n_labeled - 1, 2):
            cuts[c_id] = np.random.randint(avg_windows - min_max, avg_windows + min_max + 1)
            delta = avg_windows - cuts[c_id]
            cuts[c_id + 1] = avg_windows + delta
        
        assert np.sum(cuts) * n_files == n_labeled_windows
        cuts = list(cuts) + list(np.ones(n_clients - n_labeled, dtype=int) * -1)

        # Create dict to write as json
        federation = { 'id' + str(id): {'n_regions':int(r), 'labeled': bool(l_b)} for id, l_b, r in list(zip(client_ids, labeled.astype(np.bool), cuts))}
        federation['n_files'] = n_files
        federation['fold_folder'] = fold_folder
        federation['length'] = length
    
        federation['server'] = {'n_regions':int(-1), 'labeled': bool(False), 'n_files': server_n_files, 'length': server_length} 
    
    elif scenario == 'lab':
        cuts = np.zeros(n_labeled, dtype=int)
        for c_id in range(0, n_labeled - 1, 2):
            cuts[c_id] = np.random.randint(avg_windows - min_max, avg_windows + min_max + 1)
            delta = avg_windows - cuts[c_id]
            cuts[c_id + 1] = avg_windows + delta
        
        assert np.sum(cuts) * n_files == n_labeled_windows
        cuts = list(cuts) + list(np.ones(n_clients - n_labeled, dtype=int) * -1)

        # Create dict to write as json
        federation = { 'id' + str(id): {'n_regions':int(r), 'labeled': bool(l_b)} for id, l_b, r in list(zip(client_ids, labeled.astype(np.bool), cuts))}
        federation['n_files'] = n_files
        federation['fold_folder'] = fold_folder
        federation['length'] = length

        federation['server'] = {'n_regions':int(n_labeled_windows // 2 // server_n_files), 'labeled': bool(True), 'n_files': server_n_files, 'length': server_length} 

    elif scenario == 'las':
        # Create dict to write as json
        federation = { 'id' + str(id): {'n_regions':int(-1), 'labeled': bool(False)} for id in client_ids}
        federation['n_files'] = n_files
        federation['fold_folder'] = fold_folder
        federation['length'] = length

        federation['server'] = {'n_regions':int(n_labeled_windows // server_n_files), 'labeled': bool(True), 'n_files': server_n_files, 'length': server_length} 


    with open(fold_folder + '/dataplan_' + str(id) + '_' + scenario +'.yaml', 'w') as outfile:
        yaml.dump(federation, outfile, default_flow_style=False)
    print(federation)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--fold_folder", type=str)
    parser.add_argument("--id", type=str)
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--n_clients", type=int)
    parser.add_argument("--n_labeled_windows", type=int)
    parser.add_argument("--n_min_windows", type=int)
    parser.add_argument("--labeled_share", type=float)
    args = parser.parse_args()


    generate(fold_folder = args.fold_folder, id=args.id, scenario=args.scenario, n_clients=args.n_clients, 
                    labeled_share=args.labeled_share, n_labeled_windows=args.n_labeled_windows, n_min_windows=args.n_min_windows)