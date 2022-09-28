import argparse
import os

import scipy.stats as ss
import torch
import sys
from copy import deepcopy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch
import wandb
import numpy as np
from continuum.scenarios import ClassIncremental, ContinualScenario
from sampling import sample_classes_normal, get_probability, get_classes
from eval import probe_knn, representation_eval
from Utils import *
from utils_training import run_taskset, reset_all_weights
from Models.encoders import encoders, PreparedModel, EncoderTuple
from continuum.tasks.utils import split_train_val

def run_scenario(config):
    # cluster env      
    transformations, transformations_te = None, None
    slurm_tmpdir = os.environ.get('SLURM_TMPDIR')

    if not (slurm_tmpdir is None):
        config.root_dir = os.path.join(slurm_tmpdir, "Archives")
        config.data_dir = os.path.join(slurm_tmpdir, "Datasets")

    dataset_train, dataset_test, nb_classes, input_d, transformations, transformations_te =\
        get_dataset(config.dataset, config.data_dir, config.architecture)
    config.input_d = input_d

    if  config.dataset == "CIFAR100Lifelong":
        scenario = ContinualScenario(dataset_train, transformations=transformations)
        scenario_te = ContinualScenario(dataset_test, transformations=transformations_te)
    else:
        scenario = ClassIncremental(dataset_train, nb_tasks=nb_classes, transformations=transformations)
        scenario_te = ClassIncremental(dataset_test, nb_tasks=nb_classes, transformations=transformations_te)


    if config.num_tasks == 1:
        # we are in a iid training
        print(
            f"We are in a IID training config.num_classes = {nb_classes} and  config.classes_per_task = {nb_classes}"
        )
        config.num_classes = nb_classes
        config.classes_per_task = nb_classes

    wandb.init(
        dir=config.root_dir,
        project=config.project,
        settings=wandb.Settings(start_method="fork"),
        group="Convergence_Init",
        id="no_real_id_yet" + wandb.util.generate_id(),
        entity="clip_cl",
        notes=f"No notes yet",
        tags=[config.optim],
        config=config,
    )

    if  config.dataset == "CIFAR100Lifelong":
        full_tr_dataset = scenario[:]
        full_te_dataset = scenario_te[:]
    else:
        full_tr_dataset = scenario[: config.num_classes]
        full_te_dataset = scenario_te[: config.num_classes]


    model = get_model(config)
    opt = get_optim(model, name=config.optim, lr=config.lr, momentum=config.momentum)

    list_val_test_acc = []
    list_test_acc = []

    probability = get_probability(config)
    if config.track == "Oui":
        # init
        model.track_weight_diff()

    for task_id in range(config.num_tasks):

        if config.reinit_opt == "Yes":
            del opt
            opt = get_optim(
                model, name=config.optim, lr=config.lr, momentum=config.momentum
            )

        print(task_id)
        wandb.log({"task": task_id})
        # sample with seed for reproducibility
        # the scenario is composed of 5 binary classification classes randomly ordered

        classes = get_classes(config, task_id, probability)

        if config.debug:
            continue

        if config.dataset == "CIFAR100Lifelong":
            if config.num_tasks == 1:
                taskset_tr = full_tr_dataset
            else:
                env_id = task_id % 5
                taskset_tr = deepcopy(scenario[env_id])
                assert len(classes) == 2
                indexes = np.where((taskset_tr._y == classes[0]) | (taskset_tr._y == classes[1]))[0]
                taskset_tr._x = taskset_tr._x[indexes]
                taskset_tr._y = taskset_tr._y[indexes]
                taskset_tr._t = taskset_tr._t[indexes]
        else:
            taskset_tr = scenario[classes]
            taskset_tr,  taskset_val = split_train_val(taskset_tr, val_split = 0.1)
        if config.prob:
            taskset_te = scenario_te[classes]

        print(f"train: {classes}")
        lr_scheduler = None
        if opt is not None and config.lr_aneal:
            # reset optimizer for wach task
            opt = get_optim(model, name=config.optim, lr=config.lr, momentum=config.momentum)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.nb_epochs)
        for epoch in range(config.nb_epochs):

            if config.early_stopping != 0:
                if epoch == 0:
                    cpt = 0
                    val_acc_ES = 0
                else:
                    val_acc, val_acc_per_class = run_taskset(config, taskset_val, model, opt=None,
                                                                 nb_classes=taskset_val.nb_classes,
                                                                 batch_size=config.batch_size, masking=config.masking,
                                                                 tot_classes=config.num_classes)
                    if val_acc > val_acc_ES:
                        cpt = 0
                        val_acc_ES = val_acc
                    else:
                        cpt += 1
                        if cpt > config.early_stopping:
                            break
            train_acc, train_acc_per_class = run_taskset(config, taskset_tr, model, opt=opt, nb_classes=taskset_tr.nb_classes,
                                                         batch_size=config.batch_size, masking=config.masking, tot_classes=config.num_classes)

            if lr_scheduler is not None:
                lr_scheduler.step()

            if config.num_tasks == 1:
                test_acc, test_acc_per_class = run_taskset(config, full_te_dataset, model, opt=None,
                                                           nb_classes=full_te_dataset.nb_classes,
                                                           batch_size=config.batch_size, masking=config.masking, tot_classes=config.num_classes)
                if config.class_acc:
                    wandb.log({"train_acc": train_acc, "test_acc": test_acc, "epoch": epoch,
                               "train_acc_per_class": {str(i): acc for i, acc in enumerate(train_acc_per_class)},
                               "test_acc_per_class": {str(i): acc for i, acc in enumerate(test_acc_per_class)}})
                else:
                    wandb.log({"train_acc": train_acc, "test_acc": test_acc, "epoch": epoch})

        if config.prob:
            print("test")
            test_acc, test_acc_per_class = run_taskset(config, taskset_te, model, opt=None,
                                                       nb_classes=taskset_te.nb_classes, batch_size=config.batch_size, tot_classes=config.num_classes)
            list_test_acc.append(test_acc)

            val_test_acc = representation_eval(model, full_tr_dataset, full_te_dataset, optim_name=config.optim,
                                               nb_eval_epoch=config.nb_epoch_val,
                                               )
            list_val_test_acc.append(val_test_acc)
        else:
            print("test (full test set)")
            test_acc, test_acc_per_class = run_taskset(config, full_te_dataset, model, opt=None,
                                                       nb_classes=full_te_dataset.nb_classes,
                                                       batch_size=config.batch_size, tot_classes=config.num_classes)

        test_acc_knn = None
        if config.prob_knn:
            if task_id % config.knn_every == 0:
                test_acc_knn = probe_knn(full_tr_dataset, full_te_dataset, model, nb_classes=taskset_tr.nb_classes,
                                         batch_size=config.batch_size)
                wandb.log({'test_acc_knn': test_acc_knn}, commit=False)
                print(test_acc_knn)

        if config.prob:
            if config.class_acc:
                wandb.log({"train_acc": train_acc,
                           "test_acc": test_acc,
                           "val_test_acc": val_test_acc,
                           "train_acc_per_class": {str(i): acc for i, acc in enumerate(train_acc_per_class)},
                           "test_acc_per_class": {str(i): acc for i, acc in enumerate(test_acc_per_class)},
                           "task_index": task_id})
            else:
                wandb.log(
                    {"train_acc": train_acc, "test_acc": test_acc, "val_test_acc": val_test_acc, "task_index": task_id})
        else:
            if config.class_acc:
                wandb.log({"train_acc": train_acc, "test_acc": test_acc, "task_index": task_id, "epoch": epoch, "classes": classes,
                           "train_acc_per_class": {str(i): acc for i, acc in enumerate(train_acc_per_class)},
                           "test_acc_per_class": {str(i): acc for i, acc in enumerate(test_acc_per_class)}, })
            else:
                wandb.log({"train_acc": train_acc, "test_acc": test_acc, "task_index": task_id, "classes": classes, "epoch": epoch})
                if config.track == "Oui":
                    # init
                    diff_1, diff_2, diff_3, diff_4, diff_5 = model.track_weight_diff()
                    wandb.log(
                        {"diff_1": diff_1, "diff_2": diff_2, "diff_3": diff_3, "diff_4": diff_4, "diff_5": diff_5})



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./Archives")
    parser.add_argument("--data_dir", type=str, default="../Datasets")
    parser.add_argument("--project", type=str, default="CLONVERGENCE")
    parser.add_argument("--dataset", type=str, default="MNIST",
                        choices=["MNIST", "CIFAR10", "CIFAR100", "CUB200", "KMNIST", "fashion", "Car196", "Aircraft", "CIFAR100Lifelong"])
    parser.add_argument("--num_tasks", type=int, default=5, help="Task number")
    parser.add_argument("--num_classes", type=int, default=10, help="Task class in the full scenario")
    parser.add_argument("--model", type=str, default="baseResnet", choices=["alexnet", "resnet", "googlenet", "vgg"])
    parser.add_argument("--classes_per_task", type=int, default=2, help="number of classes wanted in each task")
    parser.add_argument("--nb_epochs", type=int, default=1, help="nb epoch to train")
    parser.add_argument("--nb_epoch_val", type=int, default=1, help="nb epoch to train probe")
    parser.add_argument("--nb_layers", type=int, default=20, help="nb epoch to train probe", choices=[20, 32, 44, 56])
    parser.add_argument("--optim", default="Adam", type=str, choices=["SGD", "Adam"])
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--entropy_decrease", default=0, type=int)
    parser.add_argument("--seed", default="1664", type=int)
    parser.add_argument("--forgetting", type=bool, default=False, help="flag to assess if forgetting still happens")
    parser.add_argument("--prob", action="store_true", default=False, help="decide if train a prob")
    parser.add_argument("--masking", default="None", type=str, choices=["None", "single", "group", "multi-head"])
    parser.add_argument("--head", default="linear", type=str, choices=["linear", "weightnorm"])
    parser.add_argument("--training", default="default", type=str, choices=["default", "incremental"])
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--early_stopping", default=0, type=int, help="early stopping criterion, if 0 no early stopping"
                                                                      " else, it design the number of epochs without"
                                                                      " progress that trigger the end of the task")
    parser.add_argument("--reinit_opt", default="No", type=str, help="Reinitialize optimizer for each task")
    parser.add_argument("--class_acc", type=bool, default=False, help="log accuracy for each class separately")
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--prob_knn', type=int, default=0)
    parser.add_argument('--knn_every', type=int, default=5)
    parser.add_argument('--uniform_range', type=int, default=8)
    parser.add_argument('--class_sampling', type=str,
                        choices=['uniform', 'gaussian', 'gaussian_with_cycles', 'uniform_with_cycles',
                                 'uniform_shifted', 'cl_with_cycles', "iid"], default='uniform')
    parser.add_argument('--cycle_size', type=int, default=100)  # every cycle_size all classes should have been seen
    parser.add_argument('--class_sampling_std', type=float, default=10)
    parser.add_argument('--lr_aneal', type=int, default=0)
    parser.add_argument('--reinit_model', type=int, default=0)
    parser.add_argument('--pretrained_model', type=str, choices=list(encoders.keys()), default=None)
    parser.add_argument('--wrn_width_factor', type=int, default=1)
    parser.add_argument('--wrn_dropout', type=float, default=0.)
    parser.add_argument('--architecture', type=str,
                        choices=['default', 'resnet', 'vgg', 'vit_b_16', 'inception', 'mobilenet_v3_small'], default='default')
    parser.add_argument("--track", default="Non", type=str, choices=["Non", "Oui"])
    parser.add_argument("--randomized_order", default="1", type=float, help="start from a fixed sequence of tasks then randomly change some classes.")
    parser.add_argument("--randomized_couples", default="1", type=float, help="define the amount of meet couples among all possible couples.")


    config = parser.parse_args()
    if "SCRATCH" in os.environ and config.data_dir == "../Datasets":
        config.data_dir = f"{os.environ.get('SCRATCH')}/Datasets/"

    if config.num_tasks > 1 and config.optim == "Adam" and config.momentum == 0:
        # adam is not controlled by momentum so this experiments does not make sens.
        sys.exit()

    if config.early_stopping != 0:
        config.nb_epochs = max(config.nb_epochs, 200)


    run_scenario(config)
