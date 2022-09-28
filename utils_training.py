import numpy as np
import tqdm
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader
import torch.nn.functional as F
from continuum.tasks import TaskType, get_balanced_sampler
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_loss(out, labels, masking, num_classes,loss_func):
    classes_mask = torch.eye(num_classes).to(device).float()
    if masking == "single":
        out = torch.mul(out, classes_mask[labels])
    elif masking == "group":
        label_unique = labels.unique()
        ind_mask = classes_mask[label_unique].sum(0)
        full_mask = ind_mask.unsqueeze(0).repeat(out.shape[0], 1)
        out = torch.mul(out, full_mask)
    loss = loss_func(out, labels.long())
    assert loss == loss, print("There should be some Nan")
    return loss

def run_taskset(config, taskset, model, opt=None, nb_classes=10, balance=False, batch_size=64, masking="None", tot_classes=10):
    vector_pred = np.zeros(0)
    vector_label = np.zeros(0)
    if opt is None:
        model.eval()
    else:
        model.train()

    sampler = None
    if balance:
        sampler = get_balanced_sampler(taskset)
    loader = DataLoader(taskset, batch_size=batch_size, sampler=sampler, shuffle= opt is not None)

    #bar = tqdm.tqdm(loader)
    for x_, y_, t_ in loader:
        x_ = x_.to(device)
        output = model(x_)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        predictions = np.array(output.max(dim=1)[1].cpu())
        vector_pred = np.concatenate([vector_pred, predictions])
        vector_label = np.concatenate([vector_label, y_.numpy()])

        loss = get_loss(output, y_.to(device), masking, tot_classes, F.cross_entropy)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            # to avoid NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if config.track == "Oui":
                grad_1, grad_2, grad_3, grad_4, grad_5 = model.track_grads()
                wandb.log({"grad_1": grad_1, "grad_2": grad_2, "grad_3": grad_3, "grad_4": grad_4, "grad_5": grad_5})
            opt.step()

    acc_per_class = np.zeros(tot_classes)
    if opt is None:
        # we log accuracy per task only for test
        for _, label in enumerate(taskset.get_classes()):
            indexes_class = np.where(vector_label == label)[0]
            classes_correctly_predicted = (vector_pred[indexes_class] == label).sum()
            acc_per_class[label] = (
                    1.0 * classes_correctly_predicted / (1.0 * len(indexes_class))
            )
    correct = (vector_pred == vector_label).sum()
    accuracy = (1.0 * correct) / len(vector_pred)

    print(f"Accuracy: {accuracy * 100} %")
    return accuracy, acc_per_class

def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)