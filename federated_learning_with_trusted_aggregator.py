pip install syft

import torch as th

import syft as sy

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
secure_worker = sy.VirtualWorker(hook, id="secure_worker")

bob.add_workers([alice,secure_worker])
alice.add_workers([bob,secure_worker])
secure_worker.add_workers([bob,alice])

data = th.tensor([[1.,1],[0,1,],[1,0],[0,0]],requires_grad=True)
target = th.tensor([[1.],[1],[0],[0]],requires_grad=True)

bob_data = data[0:2].send(bob)
bob_target = target[0:2].send(bob)

alice_data = data[2:].send(alice)
alice_target = target[2:].send(alice)

model = nn.Linear(2,1)

bobs_model = model.copy().send(bob)
alices_model = model.copy().send(alice)

bobs_opt =optim.SGD(params = bobs_model.parameters(), lr = 0.1)
alices_opt =optim.SGD(params = alices_model.parameters(), lr = 0.1)

bobs_opt.zero_grad()

bobs_pred = bobs_model(bob_data)
bobs_pred

bob_loss = ((bobs_pred-bob_target)**2).sum()

bob_loss.backward()

