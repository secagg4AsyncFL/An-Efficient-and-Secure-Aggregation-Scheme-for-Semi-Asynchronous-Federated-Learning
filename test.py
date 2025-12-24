import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.algorithm.fedprox as fedprox
import flgo.algorithm.fedavgm as fedavgm
import flgo.algorithm.secfedbuff as secfedbuff
import flgo.algorithm.pafed as pafed
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.mnist_classification.model.cnn as mnist_cnn
import flgo.benchmark.emnist_classification as emnist
import flgo.benchmark.emnist_classification.model.cnn as emnist_cnn
import flgo.benchmark.fashion_classification as fashion
import flgo.benchmark.svhn_classification as svhn
import flgo.benchmark.svhn_classification.model.cnn as svhn_cnn
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.cifar10_classification.model.cnn as cifar10_cnn
import flgo.benchmark.cifar100_classification as cifar100
import flgo.benchmark.cifar100_classification.model.cnn as cifar100_cnn
import flgo.benchmark.cifar100_classification.model.resnet18 as cifar100_resnet18

import flgo.benchmark.partition as fbp
from flgo.simulator.base import BasicSimulator



mnist_IID_task = './test_mnist_IID'
emnist_IID_task = './test_emnist_IID'
fashion_IID_task = './test_fashion_IID'
svhn_IID_task = './test_svhn_IID'
cifar10_IID_task = './test_cifar10_IID'
cifar100_IID_task = './test_cifar100_IID'

flgo.gen_task_by_(benchmark=mnist, partitioner=fbp.IIDPartitioner(num_clients=150), task_path=mnist_IID_task)
flgo.gen_task_by_(benchmark=emnist, partitioner=fbp.IIDPartitioner(num_clients=80), task_path=emnist_IID_task)
flgo.gen_task_by_(benchmark=fashion, partitioner=fbp.IIDPartitioner(num_clients=150), task_path=fashion_IID_task)
flgo.gen_task_by_(benchmark=svhn, partitioner=fbp.IIDPartitioner(num_clients=150), task_path=svhn_IID_task)
flgo.gen_task_by_(benchmark=cifar10, partitioner=fbp.IIDPartitioner(num_clients=150), task_path=cifar10_IID_task)
flgo.gen_task_by_(benchmark=cifar100, partitioner=fbp.IIDPartitioner(num_clients=150), task_path=cifar100_IID_task)



mnist_NonIID_task = './test_mnist_NonIID'
emnist_NonIID_task = './test_emnist_NonIID'
fashion_NonIID_task = './test_fashion_NonIID'
svhn_NonIID_task = './test_svhn_NonIID'
cifar10_NonIID_task = './test_cifar10_NonIID'
cifar100_NonIID_task = './test_cifar100_NonIID'

flgo.gen_task_by_(benchmark=mnist, partitioner=fbp.DirichletPartitioner(num_clients=150, alpha=0.5), task_path=mnist_NonIID_task)
flgo.gen_task_by_(benchmark=emnist, partitioner=fbp.DirichletPartitioner(num_clients=80, alpha=0.5), task_path=emnist_NonIID_task)
flgo.gen_task_by_(benchmark=fashion, partitioner=fbp.DirichletPartitioner(num_clients=150, alpha=0.5), task_path=fashion_NonIID_task)
flgo.gen_task_by_(benchmark=svhn, partitioner=fbp.DirichletPartitioner(num_clients=150, alpha=0.5), task_path=svhn_NonIID_task)
flgo.gen_task_by_(benchmark=cifar10, partitioner=fbp.DirichletPartitioner(num_clients=80, alpha=0.5), task_path=cifar10_NonIID_task)
flgo.gen_task_by_(benchmark=cifar100, partitioner=fbp.DirichletPartitioner(num_clients=80, alpha=0.5), task_path=cifar100_NonIID_task)



mnist_Div_task = './test_mnist_Div'
emnist_Div_task = './test_emnist_Div'
fashion_Div_task = './test_fashion_Div'
svhn_Div_task = './test_svhn_Div'
cifar10_Div_task = './test_cifar10_Div'
cifar100_Div_task = './test_cifar100_Div'

flgo.gen_task_by_(benchmark=mnist, partitioner=fbp.DiversityPartitioner(num_clients=150, diversity=0.5), task_path=mnist_Div_task)
flgo.gen_task_by_(benchmark=emnist, partitioner=fbp.DiversityPartitioner(num_clients=80, diversity=0.5), task_path=emnist_Div_task)
flgo.gen_task_by_(benchmark=fashion, partitioner=fbp.DiversityPartitioner(num_clients=150, diversity=0.5), task_path=fashion_Div_task)
flgo.gen_task_by_(benchmark=svhn, partitioner=fbp.DiversityPartitioner(num_clients=150, diversity=0.5), task_path=svhn_Div_task)
flgo.gen_task_by_(benchmark=cifar10, partitioner=fbp.DiversityPartitioner(num_clients=80, diversity=0.5), task_path=cifar10_Div_task)
flgo.gen_task_by_(benchmark=cifar100, partitioner=fbp.DiversityPartitioner(num_clients=80, diversity=0.5), task_path=cifar100_Div_task)




class StaticUniSimulator(BasicSimulator):
    def initialize(self):
        self.client_time_response = {cid: self.random_module.randint(600, 10800) for cid in self.clients}
        self.set_variable(list(self.clients.keys()), 'latency', list(self.client_time_response.values()))

    def update_client_responsiveness(self, client_ids):
        latency = [self.client_time_response[cid] for cid in client_ids] # 这里可以替换为其他随机数，以生成动态的响应性能异构
        self.set_variable(client_ids, 'latency', latency)






'''
+++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  PaFed-I.I.D.  ++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++
'''

secfedbuff_runner = flgo.init(task=cifar100_IID_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()



pafed_runner = flgo.init(task=cifar100_IID_task, algorithm=pafed, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0},
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=cifar10_IID_task, algorithm=pafed, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=mnist_IID_task, algorithm=pafed, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=fashion_IID_task, algorithm=pafed, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=svhn_IID_task, algorithm=pafed, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=emnist_IID_task, algorithm=pafed, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()




'''
++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++  FedAvg-I.I.D.  ++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++
'''

fedavg_runner = flgo.init(task=mnist_IID_task, algorithm=fedavg, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()
fedavg_runner = flgo.init(task=emnist_IID_task, algorithm=fedavg, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run() 
fedavg_runner = flgo.init(task=fashion_IID_task, algorithm=fedavg, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
fedavg_runner.run() 
fedavg_runner = flgo.init(task=svhn_IID_task, algorithm=fedavg,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()
fedavg_runner = flgo.init(task=cifar10_IID_task, algorithm=fedavg, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()
fedavg_runner = flgo.init(task=cifar100_IID_task, algorithm=fedavg, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0,'eta':0.5}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()



'''
++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++  FedProx-I.I.D.  ++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++
'''

fedprox_runner = flgo.init(task=mnist_IID_task, algorithm=fedprox, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()
fedprox_runner = flgo.init(task=emnist_IID_task, algorithm=fedprox, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run() 
fedprox_runner = flgo.init(task=fashion_IID_task, algorithm=fedprox, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
fedprox_runner.run() 
fedprox_runner = flgo.init(task=svhn_IID_task, algorithm=fedprox,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()
fedprox_runner = flgo.init(task=cifar10_IID_task, algorithm=fedprox , 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()
fedprox_runner = flgo.init(task=cifar100_IID_task, algorithm=fedprox, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()





'''
++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++  FedAvgm-I.I.D.  ++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++
'''

fedavgm_runner = flgo.init(task=mnist_IID_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()
fedavgm_runner = flgo.init(task=emnist_IID_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run() 
fedavgm_runner = flgo.init(task=fashion_IID_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
fedavgm_runner.run() 
fedavgm_runner = flgo.init(task=svhn_IID_task, algorithm=fedavgm,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()
fedavgm_runner = flgo.init(task=cifar10_IID_task, algorithm=fedavgm , 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()
fedavgm_runner = flgo.init(task=cifar100_IID_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()






'''
++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  secfedbuff-I.I.D.  ++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++
'''

secfedbuff_runner = flgo.init(task=mnist_IID_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()
secfedbuff_runner = flgo.init(task=emnist_IID_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run() 
secfedbuff_runner = flgo.init(task=fashion_IID_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run() 
secfedbuff_runner = flgo.init(task=svhn_IID_task, algorithm=secfedbuff,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()
secfedbuff_runner = flgo.init(task=cifar10_IID_task, algorithm=secfedbuff , 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()
secfedbuff_runner = flgo.init(task=cifar100_IID_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()





'''
++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  PaFed-NonI.I.D.  ++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++
'''

pafed_runner = flgo.init(task=cifar100_NonIID_task, algorithm=pafed, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0},
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=cifar10_NonIID_task, algorithm=pafed, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()

pafed_runner = flgo.init(task=mnist_NonIID_task, algorithm=pafed, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=emnist_NonIID_task, algorithm=pafed, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=fashion_NonIID_task, algorithm=pafed, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=svhn_NonIID_task, algorithm=pafed, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()




'''
++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  FedProx-NonI.I.D.  ++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++
'''
fedprox_runner = flgo.init(task=mnist_NonIID_task, algorithm=fedprox, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()
fedprox_runner = flgo.init(task=emnist_NonIID_task, algorithm=fedprox, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run() 
fedprox_runner = flgo.init(task=fashion_NonIID_task, algorithm=fedprox, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
fedprox_runner.run() 
fedprox_runner = flgo.init(task=svhn_NonIID_task, algorithm=fedprox,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()
fedprox_runner = flgo.init(task=cifar10_NonIID_task, algorithm=fedprox , 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()
fedprox_runner = flgo.init(task=cifar100_NonIID_task, algorithm=fedprox, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()




'''
+++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  FedAvg-NonI.I.D.  ++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++
'''

fedavg_runner = flgo.init(task=mnist_NonIID_task, algorithm=fedavg, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()
fedavg_runner = flgo.init(task=emnist_NonIID_task, algorithm=fedavg, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run() 
fedavg_runner = flgo.init(task=fashion_NonIID_task, algorithm=fedavg, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
fedavg_runner.run() 
fedavg_runner = flgo.init(task=svhn_NonIID_task, algorithm=fedavg,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()
fedavg_runner = flgo.init(task=cifar10_NonIID_task, algorithm=fedavg, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()
fedavg_runner = flgo.init(task=cifar100_NonIID_task, algorithm=fedavg, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()




'''
++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  FedAvgm-NonI.I.D.  ++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++
'''

fedavgm_runner = flgo.init(task=mnist_NonIID_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()
fedavgm_runner = flgo.init(task=emnist_NonIID_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run() 
fedavgm_runner = flgo.init(task=fashion_NonIID_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
fedavgm_runner.run() 
fedavgm_runner = flgo.init(task=svhn_NonIID_task, algorithm=fedavgm,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()
fedavgm_runner = flgo.init(task=cifar10_NonIID_task, algorithm=fedavgm , 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()
fedavgm_runner = flgo.init(task=cifar100_NonIID_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()




'''
+++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  secfedbuff-NonI.I.D.  ++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

secfedbuff_runner = flgo.init(task=mnist_NonIID_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()
secfedbuff_runner = flgo.init(task=emnist_NonIID_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run() 
secfedbuff_runner = flgo.init(task=fashion_NonIID_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run() 
secfedbuff_runner = flgo.init(task=svhn_NonIID_task, algorithm=secfedbuff,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()
secfedbuff_runner = flgo.init(task=cifar10_NonIID_task, algorithm=secfedbuff , 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()
secfedbuff_runner = flgo.init(task=cifar100_NonIID_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()




'''
++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  PaFed-Diversity  ++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++
'''
pafed_runner = flgo.init(task=cifar10_Div_task, algorithm=pafed, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=cifar100_Div_task, algorithm=pafed, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0},
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=mnist_Div_task, algorithm=pafed, 
                         option={'learning_rate':'0.1','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=emnist_Div_task, algorithm=pafed, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=fashion_Div_task, algorithm=pafed, 
                         option={'learning_rate':'0.1','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
pafed_runner.run()
pafed_runner = flgo.init(task=svhn_Div_task, algorithm=pafed, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
pafed_runner.run()




'''
++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  FedProx-Diversity  ++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++
'''

fedprox_runner = flgo.init(task=mnist_Div_task, algorithm=fedprox, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()
fedprox_runner = flgo.init(task=emnist_Div_task, algorithm=fedprox, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run() 
fedprox_runner = flgo.init(task=fashion_Div_task, algorithm=fedprox, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
fedprox_runner.run() 
fedprox_runner = flgo.init(task=svhn_Div_task, algorithm=fedprox,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()
fedprox_runner = flgo.init(task=cifar10_Div_task, algorithm=fedprox , 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()
fedprox_runner = flgo.init(task=cifar100_Div_task, algorithm=fedprox, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
fedprox_runner.run()



'''
++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  FedAvg-Diversity  ++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++
'''

fedavg_runner = flgo.init(task=mnist_Div_task, algorithm=fedavg, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()
fedavg_runner = flgo.init(task=emnist_Div_task, algorithm=fedavg, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run() 
fedavg_runner = flgo.init(task=fashion_Div_task, algorithm=fedavg, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
fedavg_runner.run() 
fedavg_runner = flgo.init(task=svhn_Div_task, algorithm=fedavg,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()
fedavg_runner = flgo.init(task=cifar10_Div_task, algorithm=fedavg, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()
fedavg_runner = flgo.init(task=cifar100_Div_task, algorithm=fedavg, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
fedavg_runner.run()



'''
++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  FedAvgm-Diversity  ++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++
'''

fedavgm_runner = flgo.init(task=mnist_Div_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()
fedavgm_runner = flgo.init(task=emnist_Div_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run() 
fedavgm_runner = flgo.init(task=fashion_Div_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
fedavgm_runner.run() 
fedavgm_runner = flgo.init(task=svhn_Div_task, algorithm=fedavgm,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()
fedavgm_runner = flgo.init(task=cifar10_Div_task, algorithm=fedavgm , 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()
fedavgm_runner = flgo.init(task=cifar100_Div_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_cnn,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()
fedavgm_runner = flgo.init(task=cifar100_Div_task, algorithm=fedavgm, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
fedavgm_runner.run()




'''
+++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++  SecFedBuff-Diversity  ++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

secfedbuff_runner = flgo.init(task=mnist_Div_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=mnist_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()
secfedbuff_runner = flgo.init(task=emnist_Div_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.02','batch_size':'512', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=emnist_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run() 
secfedbuff_runner = flgo.init(task=fashion_Div_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run() 
secfedbuff_runner = flgo.init(task=svhn_Div_task, algorithm=secfedbuff,     
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0}, 
                         model=svhn_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()
secfedbuff_runner = flgo.init(task=cifar10_Div_task, algorithm=secfedbuff , 
                         option={'learning_rate':'0.05','batch_size':'64', 'sample':'uniform', 'num_rounds':100, 'num_epochs':5, 'gpu':0},
                         model=cifar10_cnn,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()

secfedbuff_runner = flgo.init(task=cifar100_Div_task, algorithm=secfedbuff, 
                         option={'learning_rate':'0.01','batch_size':'64', 'sample':'uniform', 'num_rounds':200, 'num_epochs':5, 'gpu':0}, 
                         model=cifar100_resnet18,
                         Simulator=StaticUniSimulator)
secfedbuff_runner.run()