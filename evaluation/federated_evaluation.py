import numpy as np
from single.utils import bcolors
from run_single import run
from copy import deepcopy

class FederatedEvaluator():
    def __init__(self, clients, server, cfg, verbose=0):
        self.clients = clients
        self.server = server
        self.cfg = cfg
        self.verbose=verbose

    def evaluate(self, name='', clients=False, server=True):
        print(bcolors.WARNING + '\n\n' + ("+" * 80) + "\n" + "Evaluate Federated Learning" + ("\n") + ("+" * 80) + bcolors.ENDC)    
        if clients:
            self._evaluate_clients()
        if server:
            cfg = deepcopy(self.server)
            cfg['result_file'] = self.server['result_file'] + str(name) + '.json'
            self._evaluate_server(cfg)

    def _evaluate_clients(self):
        print(bcolors.FAIL + "\n**** Evaluate..." + bcolors.FAIL)  
        for client_id in np.arange(len(self.clients)):
            print(bcolors.FAIL + "**  ...Client " + str(self.clients[client_id]['task']['id']) + bcolors.ENDC)    
            run(self.clients[client_id], verbose=self.verbose)

    def _evaluate_server(self, cfg):
        print(bcolors.FAIL + "\n**** Evaluate Server" + bcolors.FAIL)    
        run(cfg, verbose=self.verbose)
