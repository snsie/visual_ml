# import sys
# sys.path.append('../')

from torchdyn.models import *
from torchdyn.datasets import *
from torchdyn import *
import numpy as np
import torch
import json
from json import JSONEncoder

from torchdyn.models import *
from torchdyn.datasets import *
from torchdyn import *
import numpy as np
import torch
import json
from json import JSONEncoder
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.utils.data as data

# Generate 3D nested spheres data
d = ToyDataset()
X, yn = d.generate(n_samples=2 << 12, dataset_type='spheres')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data into dataloader
bs = len(X)
# CELL
X_train = torch.Tensor(X).to(device)
y_train = torch.LongTensor(yn.long()).to(device)
train = data.TensorDataset(X_train, y_train)
trainloader = data.DataLoader(train, batch_size=bs, shuffle=True)
# CELL
import torch.nn as nn
import pytorch_lightning as pl


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        # JSONEncoder.FLOAT_REPR = lambda o: format(o, '.2f')
        if isinstance(obj, np.ndarray):
            out = obj.tolist()

            # out=np.round(out,5)
            for i, iarray in enumerate(out):
                if isinstance(iarray, list):
                    out[i] = np.around(iarray, 5)
                    # out[i] = [f"{num:.5f}" for num in iarray]
                else:
                    out[i] = np.round(iarray, 5)
            # if isinstance(out[0],list)==False:
                # print('hello')
                # out=np.around(out,5)

            return out

            # print(i)
            # print('done')
            # return out
        return JSONEncoder.default(self, obj)


class Learner(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return trainloader


# We consider 5 augmenting dimensions, i.e. the DEFunc must accomodate 8 inputs
func = nn.Sequential(nn.Linear(6, 64),
                     nn.Tanh(),
                     nn.Linear(64, 3)
                     )


# Define NeuralDE
neuralDE = NeuralDE(func, solver='dopri5', order=2).to(device)

# Here we specify to the "Augmenter" the 5 extra dims. For 0-augmentation, we do not need to pass additional arg.s
model = nn.Sequential(
    Augmenter(augment_dims=3),
    neuralDE,
    nn.Linear(6, 2)).to(device)
# Train the model
epoch_checkpts = [1, 1, 3, 5, 10, 30, 50]
curr_epoch = 200

# for i in epoch_checkpts:
# curr_epoch += i
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='./data/ho_chpts/',
    filename='{epoch:04d}-{loss:.3f}.hdf5',
    period=1,
    monitor='train_loss',
    # save_weights_only=True,
    save_top_k=curr_epoch,
    prefix='model_chpt'
)
trainer = pl.Trainer(max_epochs=curr_epoch,
                     checkpoint_callback=checkpoint_callback)
learn = Learner(model)
num = str(curr_epoch).zfill(4)
fname = "./data/numpyData_e_{}.json".format(num)

trainer.fit(learn)

print("done")

model_weights = []
numpyData = {}
for i, param in enumerate(model.parameters()):
    temp_array = param.detach().cpu().numpy()
    # temp_array=temp_array.astype(np.half)
    # if i==0:
    # print(np.size(temp_array))
    layerName = 'layer_' + str(int(i / 2) + 1)

    if i % 2 == 0:
        if layerName not in numpyData:
            arrayShape = np.shape(temp_array)
            weightCount = arrayShape[0] * arrayShape[1]
            biasCount = arrayShape[0]
            numpyData[layerName] = {
                'weightCount': weightCount, 'biasCount': biasCount}
            if i == 0:
                numpyData[layerName]['actFunc'] = 'Tanh'
            else:
                numpyData[layerName]['actFunc'] = 'Linear'
        varName = 'weight'
    else:
        varName = 'bias'
        # print(np.size(temp_array))
    numpyData[layerName][varName] = temp_array
    # file_name='../data_out/weights_'+str(i)+'.txt'
    # np.savetxt(file_name,temp_array, fmt='%1.5f')
with open(fname, "w") as write_file:
    json.dump(numpyData, write_file, cls=NumpyArrayEncoder)


# # c = ['blue', 'orange']
# # fig = plt.figure(figsize=(6, 6))
# # ax = fig.add_subplot(111, projection='3d')
# # for i in range(2):
# #     ax.scatter(X[yn == i, 0], X[yn == i, 1],
# #                X[yn == i, 2], s=5, alpha=0.5, c=c[i])
# # plt.show()
# # a = np.arry()
a = np.array(2)
b = np.array(3)
