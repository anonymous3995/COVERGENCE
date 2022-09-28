import torch
from torchvision import transforms
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch.nn.functional as F
import torch.optim as optim
from continuum.datasets import MNIST, InMemoryDataset
from continuum.tasks import TaskType, get_balanced_sampler
from continuum.datasets import CIFAR10, CIFAR100, KMNIST, MNIST, FashionMNIST, CUB200, Car196, FGVCAircraft

from Models.model import Classifier, Model, get_CIFAR_Model, EncoderClassifier
from Models.encoders import encoders, PreparedModel, EncoderTuple
from utils_training import reset_all_weights




def get_dataset(dataset_name, data_dir, architecture="default"):
    transformations = None
    transformations_te = None
    if dataset_name == "MNIST":
        dataset_train = MNIST(data_dir, train=True)
        dataset_test = MNIST(data_dir, train=False)
        nb_classes = 10
        input_d = 28
    elif dataset_name == "fashion":
        dataset_train = FashionMNIST(data_dir, train=True)
        dataset_test = FashionMNIST(data_dir, train=False)
        nb_classes = 10
        input_d = 28
    elif dataset_name == "KMNIST":
        dataset_train = KMNIST(data_dir, train=True)
        dataset_test = KMNIST(data_dir, train=False)
        nb_classes = 10
        input_d = 28
    elif dataset_name == "CUB200":
        dataset_train = CUB200(data_dir, train=True)
        dataset_test = CUB200(data_dir, train=False)
        nb_classes = 200
        input_d = 100
        horizontal_flip = 0.5
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size = [3, 100, 100]
        # use continuum transfroms
        transformations = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transformations_te = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
    elif dataset_name == "Car196":
        dataset_train = Car196(data_dir, train=True)
        dataset_test = Car196(data_dir, train=False)
        nb_classes = 196
        input_d = 100
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size = [3, 100, 100]
        # use continuum transfroms
        transformations = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        # transforms.Normalize(mean, std)]

        transformations_te = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

    elif dataset_name == "Aircraft":
        dataset_train = FGVCAircraft(data_dir, train=True)
        dataset_test = FGVCAircraft(data_dir, train=False)
        nb_classes = 100
        input_d = 100
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size = [3, 100, 100]
        # use continuum transfroms
        transformations = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        # transforms.Normalize(mean, std)]

        transformations_te = [
            transforms.Resize([size[-1], size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
    elif dataset_name == "CIFAR100":
        dataset_train = CIFAR100(data_dir, train=True)
        dataset_test = CIFAR100(data_dir, train=False)
        nb_classes = 100
        input_d = 32
    elif dataset_name == "CIFAR100Lifelong":
        dataset_train = CIFAR100(data_dir, train=True, labels_type="category", task_labels="lifelong")
        dataset_test = CIFAR100(data_dir, train=False, labels_type="category", task_labels="lifelong")
        nb_classes = 20
        input_d = 32
    else:
        dataset_train = CIFAR10(data_dir, train=True)
        dataset_test = CIFAR10(data_dir, train=False)
        nb_classes = 10
        input_d = 32

    if architecture != "default":
        size = 224
        if architecture == "inception":
            size = 299
        transformations = [transforms.Resize((size, size)),transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        transformations_te = transformations
    return dataset_train, dataset_test, nb_classes, input_d, transformations, transformations_te

def get_optim(model, name, lr, momentum):
    if name == "SGD":
        opt = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
    else:
        opt = optim.Adam(params=model.parameters(), lr=lr)
    return opt


def get_model(config):

    if config.pretrained_model is None:
        if config.dataset in ["MNIST", 'mnist_fellowship', 'fashion', 'KMNIST']:
            model = Model(head_name=config.head, masking=config.masking).to(device)
        else:
            model = get_CIFAR_Model(config, num_classes=config.num_classes, head_name=config.head,
                               masking=config.masking, nb_layers=config.nb_layers, ).to(device)
    else:
        # This does not work with MNIST
        encoder_tuple: EncoderTuple = encoders[config.pretrained_model]
        encoder: PreparedModel = encoder_tuple.partial_encoder(device=device, input_shape=config.input_d,
                                                               fix_batchnorms_encoder=False,
                                                               width_factor=config.wrn_width_factor,
                                                               droprate=config.wrn_dropout)
        tr, tr_te = encoder.transformation, encoder.transformation_val
        if tr is not None:
            transformations = tr
            if tr_te is not None:
                transformations_te = tr_te
            else:
                transformations_te = tr
        classifier = encoder.classifier
        if classifier is None:
            classifier = Classifier(num_classes=config.num_classes, in_d=encoder.latent_dim, head_name=config.head,
                                    masking=config.masking).to(device)
        model = EncoderClassifier(encoder=encoder.encoder, classifier=classifier).to(device)
        if config.reinit_model:
            reset_all_weights(model)
    return model




def init_state_dict(config, taskset):
    dict_state = {}
    size = len(np.where(taskset._t >= 0)[0])

    # taskset sanity check (we want all no negative indexes to be in the start by order)
    assert np.all(np.arange(size) == taskset._t[:size])

    dict_state["scores"] = np.zeros(size)
    if config.selection == "forgetting":
        dict_state["last_pred"] = np.zeros(size)

    if config.integration:
        dict_state["nb_updates"] = np.zeros(size)
    return dict_state


def run_taskset_selection(
    config,
    taskset,
    model,
    epoch,
    opt=None,
    balance=False,
    dict_state=None,
    compute_scores=False,
):
    # init dictionnary to keep useful information
    if compute_scores and dict_state is None:
        dict_state = init_state_dict(config, taskset)

    vector_pred = np.zeros(0)
    vector_label = np.zeros(0)

    if opt is None:
        model.eval()

    if balance:
        # balance sampling of data based on label
        sampler = get_balanced_sampler(taskset)
        dataloader = DataLoader(taskset, batch_size=config.batchsize, sampler=sampler)
    else:
        dataloader = DataLoader(taskset, batch_size=config.batchsize, shuffle=True)
    for x_, y_, id_ in dataloader:
        x_ = x_.to(device)
        output = model(x_)
        predictions = np.array(output.max(dim=1)[1].cpu())
        vector_pred = np.concatenate([vector_pred, predictions])
        vector_label = np.concatenate([vector_label, y_.view(-1).numpy()])
        loss = F.cross_entropy(output, y_.to(device))

        if opt is not None:
            if compute_scores:
                dict_state = get_scores(
                    config, output, y_, id_.numpy(), epoch, dict_state=dict_state
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

    correct = (vector_pred == vector_label).sum()
    accuracy = (1.0 * correct) / len(vector_pred)
    if opt is None:
        print(f"Test Accuracy: {accuracy * 100} %")
    else:
        print(f"Train Accuracy: {accuracy * 100} %")

    return accuracy, dict_state


# def integrate_scores(historic_scores, new_scores):
#     """take into account the historic of scores to update score and make better decision."""
#
#     scores = new_scores[0]
#     idxs = new_scores[1]
#
#     # reordering
#     ord_ids = np.argsort(idxs)
#
#     scores = scores[ord_ids]
#     idxs = idxs[ord_ids]
#
#     if historic_scores is None:
#         historic_scores = {"scores": scores, "indexes": idxs, "weight": 1}
#     else:
#         assert np.all(historic_scores["indexes"] == idxs)
#
#         # homogeneous to the mean
#         historic_scores["scores"] += scores
#         # weight is useless for now but maybe later it will be useful
#         historic_scores["weight"] += 1
#
#     return historic_scores, new_scores


def get_selection(dataset, state_dict, criterion="max", probability=0.1):

    # scores are already correctly ordered theoritically.
    scores = state_dict["scores"]

    # first remove ids < 0 (it marks data that we do not want to keep)
    # ids2keep = np.where(ids > 0)[0]
    # ids = ids[ids2keep]
    # scores = scores[ids2keep]

    # size selection
    size_selection = int(len(scores) * probability)

    # get indexes from the smallest score to the biggest
    if criterion == "random":
        ids_scores = torch.randperm(len(scores))[:size_selection].numpy()
    elif criterion == "max":
        ord_ids_scores = np.argsort(scores)
        ids_scores = ord_ids_scores[-size_selection:]
    elif criterion == "min":
        ord_ids_scores = np.argsort(scores)
        ids_scores = ord_ids_scores[:size_selection]
    elif criterion == "ext":
        ord_ids_scores = np.argsort(scores)
        # half samples are in the beginning half are in the end.
        mid_1 = int(size_selection // 2)
        mid_2 = size_selection - mid_1
        ids_scores = np.concatenate([ord_ids_scores[:mid_1], ord_ids_scores[-mid_2:]])
    elif criterion == "mean":
        mean_score = scores.mean()
        # select samples the closest to the mean
        ids_scores = np.argsort(scores - mean_score)[:size_selection]
    elif criterion == "median":
        median_score = scores.median()
        # select samples the closest to the median
        ids_scores = np.argsort(scores - median_score)[:size_selection]
    else:
        raise NotImplementedError(f"criterion: {criterion} not implemented")

    ids_selection = ids_scores.astype(int)
    x, y, t = dataset.get_raw_samples(ids_selection)

    # selected_dataset = TensorDataset(selection[0], selection[1])
    return InMemoryDataset(x, y, t, TaskType.IMAGE_ARRAY).to_taskset()
