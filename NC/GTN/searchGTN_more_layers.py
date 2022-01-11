


import time
import subprocess
import multiprocessing
from threading import main_thread

class Run( multiprocessing.Process):
    def __init__(self,command):
        super().__init__()
        self.command=command
    def run(self):
        
        subprocess.run(self.command,shell=True)

dataset_to_evaluate=[("DBLP",2),("ACM",3),("IMDB",4)]
gpus=["0","1"]
total_trial_num=50

for dataset,worker_num in dataset_to_evaluate:
    for layers in [4,8,12,16]:
        study_name=f"layer_{layers}_GTN_{dataset}"
        study_storage=f"sqlite:///{study_name}.db"
        trial_num=int(total_trial_num/ (len(gpus)*worker_num) )

        process_queue=[]
        for gpu in gpus:
            for _ in range(worker_num):
                command=f"python main_sparse.py --GNN GTN --num_layers {layers} --dataset {dataset} --gpu {gpu} --trial_num {trial_num} --study_name {study_name} --study_storage {study_storage}"
                p=Run(command)
                p.daemon=True
                p.start()
                process_queue.append(p)
                time.sleep(5)

        for p in process_queue:
            p.join()






