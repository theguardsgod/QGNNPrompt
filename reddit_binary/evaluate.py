import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    dataset,
    model,
    folds,
    epochs,
    batch_size,
    lr,
    lr_decay_factor,
    lr_decay_step_size,
    weight_decay,
    use_tqdm=True,
    writer=None,
    logger=None,
    saves=None,
):

    val_losses, accs, durations = [], [], []

    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if "adj" in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        #model.to(device).reset_parameters()
       
        state_dict = torch.load(f"./saves/GIN_int4_fold-{fold}.pth")
        # Iterate through state_dict to find mismatched shapes
        for key, value in state_dict.items():
            if value.shape == torch.Size([0]):  # If it is an empty tensor
                print(f"Fixing tensor {key} with shape {value.shape}")
                state_dict[key] = torch.tensor([])  # Replace with scalar tensor

        # Move the model to the specified device
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        if use_tqdm:
            t = tqdm(total=epochs, desc="Fold #" + str(fold))
        
        max_acc = 0
        accs.append(eval_acc(model, test_loader))

        

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        durations.append(t_end - t_start)


    acc = accs
    duration = tensor(durations)
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print(
        "Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}".format(
             acc_mean, acc_std, duration_mean
        )
    )


    return acc_mean, acc_std


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction="sum").item()
    return loss / len(loader.dataset)
