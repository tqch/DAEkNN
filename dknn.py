import torch
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pgd import PGD


class DKNN:

    def __init__(
        self,
        model,
        train_data,
        train_targets,
        n_class=10,
        hidden_layers=-1,
        n_neighbors=5,
        metric="l2",
        device=torch.device("cpu")
    ):

        self.hidden_layers = hidden_layers
        self._model = self._wrap_model(model)

        
        self.train_data = train_data
        self.train_targets = np.array(train_targets)

        self.hidden_layers = self._model.hidden_layers

        self.device = device
        self._model.eval()  # make sure the model is in the eval mode
        self._model.to(self.device)

        self.n_class = n_class
        self.metric = metric
        self.n_neighbors = n_neighbors
        self._nns = self._build_nns()

    def _get_hidden_repr(self, x, batch_size=128, return_targets=False):
        hidden_reprs = []
        targets = None
        if return_targets:
            outs = []

        for i in range(0, x.size(0), batch_size):
            x_batch = x[i:i + batch_size]
            if return_targets:
                hidden_reprs_batch, outs_batch = self._model(x_batch.to(self.device))
            else:
                hidden_reprs_batch, _ = self._model(x_batch.to(self.device))
            if self.metric == "cosine":
                hidden_reprs_batch = [
                    hidden_repr_batch/hidden_repr_batch.pow(2).sum(dim=1,keepdim=True).sqrt()
                    for hidden_repr_batch in hidden_reprs_batch
                ]
            hidden_reprs.append(hidden_reprs_batch)
            if return_targets:
                outs.append(outs_batch)

        hidden_reprs = [
            np.concatenate([hidden_batch[i] for hidden_batch in hidden_reprs], axis=0)
            for i in range(len(self.hidden_layers))
        ]

        if return_targets:
            outs = np.concatenate(outs, axis=0)
            targets = outs.argmax(axis=1)

        return hidden_reprs, targets

    def _wrap_model(self, model):

        class ModelWrapper(nn.Module):

            def __init__(self, model, hidden_layers):
                super(ModelWrapper, self).__init__()
                self._model = model
                self.hidden_mappings = [
                    m[1] for m in model.named_children()
                    if isinstance(m[1], nn.Sequential) and "classifier" not in m[0]
                ]
                if hidden_layers == -1:
                    self.hidden_layers = list(range(len(self.hidden_mappings)))
                else:
                    self.hidden_layers = hidden_layers
                self.classifier = self._model.classifier

            def forward(self, x):
                hidden_reprs = []
                for mp in self.hidden_mappings:
                    x = mp(x)
                    hidden_reprs.append(x.detach().cpu())
                out = self.classifier(x.flatten(start_dim=1) if x.ndim!=2 else x)
                return [hidden_reprs[i].flatten(start_dim=1) for i in self.hidden_layers], out

        return ModelWrapper(model, self.hidden_layers)

    def _build_nns(self):
        hidden_reprs, _ = self._get_hidden_repr(self.train_data)
        return [
            NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(hidden_repr)
            for hidden_repr in tqdm(hidden_reprs)
        ]

    def predict(self, x):
        hidden_reprs, _ = self._get_hidden_repr(x)
        nn_indices = [
            nn.kneighbors(hidden_repr, return_distance=False)
            for nn, hidden_repr in zip(self._nns, hidden_reprs)
        ]
        nn_indices = np.concatenate(nn_indices, axis=1)
        nn_labels = self.train_targets[nn_indices]
        nn_labels_count = np.stack(list(map(
            lambda x: np.bincount(x, minlength=10),
            nn_labels
        )))
        return nn_labels_count/len(self.hidden_layers)/self.n_neighbors

    def __call__(self, x):
        return self.predict(x)

    
class AdvDKNN:

    def __init__(
        self,
        model,
        train_data,
        train_targets,
        attack=PGD,
        n_class=10,
        hidden_layers=-1,
        n_neighbors=5,
        device=torch.device("cpu"),
        weight=None,
        **attack_params
    ):

        self.hidden_layers = hidden_layers
        self._model = self._wrap_model(model)
        self.attack = attack(**attack_params)
        self.train_data = train_data
        self.train_targets = train_targets
           
        self.train_data_adv = self.attack.generate(self._model,self.train_data,self.train_targets,device=device)
        
        self.hidden_layers = self._model.hidden_layers

        self.device = device
        self._model.eval()  # make sure the model is in the eval mode
        self._model.to(self.device)

        self.n_class = n_class
        self.n_neighbors = n_neighbors
        self._nns = self._build_nns()
        
        self.weight = weight

    def _get_hidden_repr(self, x, batch_size=128, return_targets=False):

        hidden_reprs = []
        targets = None
        if return_targets:
            outs = []

        for i in range(0, x.size(0), batch_size):
            x_batch = x[i:i + batch_size]
            if return_targets:
                hidden_reprs_batch, outs_batch = self._model(x_batch.to(self.device))
            else:
                hidden_reprs_batch, _ = self._model(x_batch.to(self.device))
            hidden_reprs.append(hidden_reprs_batch)
            if return_targets:
                outs.append(outs_batch)

        hidden_reprs = [
            np.concatenate([hidden_batch[i].flatten(start_dim=1) for hidden_batch in hidden_reprs], axis=0)
            for i in range(len(self.hidden_layers))
        ]

        if return_targets:
            outs = np.concatenate(outs, axis=0)
            targets = outs.argmax(axis=1)

        return hidden_reprs, targets

    def _wrap_model(self, model):

        class ModelWrapper(nn.Module):

            def __init__(self, model, hidden_layers):
                super(ModelWrapper, self).__init__()
                self._model = model
                self.hidden_mappings = [
                    m[1] for m in model.named_children()
                    if isinstance(m[1], nn.Sequential) and "classifier" not in m[0]
                ]
                if hidden_layers == -1:
                    self.hidden_layers = list(range(len(self.hidden_mappings)))
                else:
                    self.hidden_layers = hidden_layers
                self.classifier = self._model.classifier

            def forward(self, x):
                hidden_reprs = []
                for mp in self.hidden_mappings:
                    x = mp(x)
                    hidden_reprs.append(x.detach().cpu())
                out = self.classifier(x.flatten(start_dim=1))
                return [hidden_reprs[i] for i in self.hidden_layers], out

        return ModelWrapper(model, self.hidden_layers)

    def _build_nns(self):
        hidden_reprs, _ = self._get_hidden_repr(self.train_data)
        hidden_reprs_adv, _ = self._get_hidden_repr(self.train_data_adv)
        return [
            NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(hidden_repr)
            for hidden_repr in tqdm(hidden_reprs)
        ],[
            NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1).fit(hidden_repr_adv)
            for hidden_repr_adv in tqdm(hidden_reprs_adv)
        ]

    def predict(self, x):
        hidden_reprs, _ = self._get_hidden_repr(x)
        nn_indices = [
            nn.kneighbors(hidden_repr)
            for nn, hidden_repr in zip(self._nns[0], hidden_reprs)
        ]
        nn_dists = [nn_inds[0].mean(axis=1) for nn_inds in nn_indices]
        nn_indices = [nn_inds[1] for nn_inds in nn_indices]
        nn_labels = [self.train_targets[nn_inds] for nn_inds in nn_indices]
        nn_labels_count = [
            np.stack(list(map(
                lambda x: np.bincount(x, minlength=10),
                nn_labs
            ))) for nn_labs in nn_labels
        ]
        nn_indices_adv = [
            nn.kneighbors(hidden_repr)
            for nn, hidden_repr in zip(self._nns[1], hidden_reprs)
        ]
        nn_dists_adv = [nn_inds_adv[0].mean(axis=1) for nn_inds_adv in nn_indices_adv]
        nn_indices_adv = [nn_inds_adv[1] for nn_inds_adv in nn_indices_adv]
        nn_labels_adv = [self.train_targets[nn_inds_adv] for nn_inds_adv in nn_indices_adv]
        nn_labels_count_adv = [
            np.stack(list(map(
                lambda x: np.bincount(x, minlength=10),
                nn_labs_adv
            ))) for nn_labs_adv in nn_labels_adv
        ]
        if self.weight is None:
            weights = [1/(1+np.exp(dist_adv-dist))[:,None] for dist,dist_adv in zip(nn_dists,nn_dists_adv)]
        else:
            weights = self.weight
        prediction = [
            wgts*nn_labs_count+(1-wgts)*nn_labs_count_adv
            for wgts,nn_labs_count,nn_labs_count_adv in zip(
                weights,nn_labels_count,nn_labels_count_adv
            )
        ]
        prediction = np.stack(prediction, axis=0).sum(axis=0)/self.n_neighbors/len(self.hidden_layers)
        return prediction

    def __call__(self, x):
        return self.predict(x)
    

if __name__ == "__main__":
    import os
    from attacks.pgd import PGD
    from models.vgg import VGG16
    from torchvision.datasets import CIFAR10

    ROOT = "./datasets"
    MODEL_WEIGHTS = "./model_weights/cifar_vggrob.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DOWNLOAD = not os.path.exists("../datasets/cifar-10-python.tar.gz")

    trainset = CIFAR10(root=ROOT, train=True, download=DOWNLOAD)
    testset = CIFAR10(root=ROOT, train=False, download=DOWNLOAD)
    train_data, train_targets = (
        torch.FloatTensor(trainset.data.transpose(0, 3, 1, 2) / 255.)[:2000],
        torch.LongTensor(trainset.targets)[:2000]
    )  # for memory's sake, only take 2000 as train set

    model = VGG16()
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    dknn = DKNN(model, train_data, train_targets, device=DEVICE)

    pgd = PGD(eps=8 / 255., step_size=2 / 255., batch_size=128)
    x, y = (
        torch.FloatTensor(testset.data.transpose(0, 3, 1, 2) / 255.)[:256],
        torch.LongTensor(testset.targets)[:256]
    )  # for memory's sake, only take 2000 as train set

    x_adv = pgd.generate(model, x, y, device=DEVICE)

    pred_benign = dknn(x.to(DEVICE)).argmax(axis=1)
    acc_benign = (pred_benign == y.numpy()).astype("float").mean()
    print(f"The benign accuracy is {acc_benign}")

    pred_adv = dknn(x_adv.to(DEVICE)).argmax(axis=1)
    acc_adv = (pred_adv == y.numpy()).astype("float").mean()
    print(f"The adversarial accuracy is {acc_adv}")
