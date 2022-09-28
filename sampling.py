import torch
import scipy.stats as ss
import numpy as np
import wandb
import itertools

import matplotlib.pyplot as plt


def sample_classes_normal(mean, std, range=[0, 100], size=2, replace=True):
    # # std = np.sqrt(var)
    # myclip_a=range[0]
    # myclip_b=range[1]
    # a, b = (myclip_a - mean) / std, (myclip_b - mean) / std
    # # X = truncnorm(a=-range/std, b=+range/std, scale=std).rvs(size=size)
    # X = truncnorm.rvs(a,b, mean,std, size=size)
    # X = X.round().astype(int)#+mean
    a = range[0]
    b = range[1]
    x = np.arange(a, b)
    # std=100
    # mean=50
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, loc=mean, scale=std) - ss.norm.cdf(xL, loc=mean, scale=std)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size=size, p=prob, replace=replace)
    # plt.hist(nums, bins = len(x))
    return nums


def get_probability(config):
    probability = np.ones(config.num_classes) / config.num_classes
    if config.entropy_decrease != 0:
        probability = (
                probability
                - (1 / config.num_classes) * np.arange(config.num_classes) * 0.05
        )
        probability /= probability.sum()
        probability = (
                probability ** config.entropy_decrease
                / (probability ** config.entropy_decrease).sum()
        )
        np.random.seed(config.seed)
        np.random.shuffle(probability)
    return probability


def get_classes(config, task_id, probability):
    n_cycles = int(config.num_tasks / config.cycle_size)
    cycle_idx = 0
    task_id_in_cycle = 0
    max_ = 0

    if config.randomized_order != 1:
        # randomized order parameter assess how the structure of the stream impact results
        list_couples = list(itertools.combinations(np.arange(config.num_classes), config.classes_per_task))
        nb_couples = len(list_couples)
        index = task_id % nb_couples
        classes = np.array(list_couples[index])

        # change class value with probability config.randomized_order
        id_change_classes = \
        np.where(np.random.randint(0, 99, size=config.classes_per_task) < config.randomized_order * 100)[0]
        if len(id_change_classes) > 0:  # if somme classes need to be changed
            # create random news classes
            new_classes = np.random.randint(0, config.num_classes, size=config.classes_per_task)
            # change the classes selected
            classes[id_change_classes] = new_classes[id_change_classes]
    elif config.randomized_couples != 1:
        if task_id == 0:
            # we set the couples of possible classes
            # it is useful to measure how important it is that all classes meet all classes within a task.
            list_couples = np.array(
                list(itertools.combinations(np.arange(config.num_classes), config.classes_per_task)))

            # shuffle along first dimension (it does not change the unique combinations)
            np.random.shuffle(list_couples)

            # config.randomized_couples is a pourcentage
            nb_select = int(config.randomized_couples * len(list_couples))

            # set static variable
            get_classes.couple_selected = list_couples[:nb_select, :]

        idx = np.random.randint(0, len(get_classes.couple_selected))
        classes = get_classes.couple_selected[idx]

    else:
        if config.training == "incremental":
            # if incremental we create 5 long term data distribution
            num_period = 2
            period_size = config.num_tasks // num_period
            nb_classes_per_period = config.num_classes // num_period
            assert nb_classes_per_period * num_period == config.num_classes, print("we will not see all classes")
            assert nb_classes_per_period >= config.classes_per_task, print("we need more classes per periode")
            start_class = (task_id // period_size) * nb_classes_per_period
            end_class = (1 + (task_id // period_size)) * nb_classes_per_period
            list_classes = np.arange(config.num_classes)[start_class:end_class]
            local_probability = probability[start_class:end_class]
            local_probability /= local_probability.sum()  # normalization
            classes = np.random.choice(
                list_classes, p=local_probability, size=config.classes_per_task, replace=False
            )

        else:
            if config.forgetting and task_id > config.num_tasks / 2:
                assert config.entropy_decrease == 0, print(
                    "there is no experiments combining entropy decrease and forgetting"
                )
                # we remove half the classes to assess if there is forgetting
                print("forgetting will start")
                classes = list(torch.randperm(config.num_classes // 2)[:config.classes_per_task].long())
            else:
                if config.class_sampling == 'uniform':
                    classes = np.random.choice(np.arange(config.num_classes), p=probability,
                                               size=config.classes_per_task, replace=False)
                elif config.class_sampling == 'iid':
                    classes = np.arange(config.num_classes)
                elif config.class_sampling == 'uniform_shifted':
                    range_ = config.num_classes
                    mean = int((range_ / config.num_tasks) * task_id)
                    min_ = max(mean - int(config.uniform_range / 2), 0)
                    max_ = min(mean + int(config.uniform_range / 2), config.num_classes)
                    classes_list = np.arange(min_, max_)
                    wandb.log({"class_distribution_mean": mean})
                    wandb.log({"class_distribution_min": min_})
                    wandb.log({"class_distribution_max": max_})
                    classes = np.random.choice(classes_list, size=config.classes_per_task, replace=False)

                elif config.class_sampling == 'gaussian':
                    # Gaussian with varying std.
                    # mean shifts
                    # sample classes from shifting gaussian
                    # save class distribution
                    range_ = config.num_classes
                    mean = int((range_ / config.num_tasks) * task_id)
                    X = sample_classes_normal(mean, config.class_sampling_std, [0, config.num_classes], size=1000,
                                              replace=True)
                    plt.clf()
                    fig = plt.figure(figsize=(8, 8))
                    plt.hist(X, 100)
                    wandb.log({"class_distribution": wandb.Image(fig)})
                    wandb.log({"class_distribution_mean": mean})
                    plt.close(fig)
                    classes = sample_classes_normal(mean, config.class_sampling_std, [0, config.num_classes],
                                                    size=config.classes_per_task,
                                                    replace=False)  # mean,config.class_sampling_std, int(config.num_classes),config.classes_per_task)

                elif config.class_sampling == 'gaussian_with_cycles':
                    wandb.log({"cycle_idx": cycle_idx})
                    wandb.log({"task_id_in_cycle": task_id_in_cycle})
                    if task_id % config.cycle_size == 0 and task_id > 0:
                        cycle_idx += 1
                        task_id_in_cycle = 0
                    # gaussian with cycles
                    range_ = config.num_classes
                    mean = int((range_ / config.cycle_size) * task_id_in_cycle)
                    X = sample_classes_normal(mean, config.class_sampling_std, [0, config.num_classes], size=1000,
                                              replace=True)
                    plt.clf()
                    fig = plt.figure(figsize=(8, 8))
                    wandb.log({"class_distribution": wandb.Image(fig)})
                    wandb.log({"class_distribution_mean": mean})
                    plt.close(fig)
                    classes = sample_classes_normal(mean, config.class_sampling_std, [0, config.num_classes],
                                                    size=config.classes_per_task,
                                                    replace=False)  # mean,config.class_sampling_std, int(config.num_classes),config.classes_per_task)
                    task_id_in_cycle += 1

                elif config.class_sampling == 'uniform_with_cycles':
                    wandb.log({"cycle_idx": cycle_idx})
                    wandb.log({"task_id_in_cycle": task_id_in_cycle})
                    if task_id % config.cycle_size == 0 and task_id > 0:
                        cycle_idx += 1
                        task_id_in_cycle = 0
                    range_ = config.num_classes
                    min_ = int((range_ / config.cycle_size) * task_id_in_cycle)
                    # min_ = max(mean-int(config.uniform_range/2),0)
                    # max_ = min(mean+int(config.uniform_range/2),config.num_classes)
                    # min_ = task_id_in_cycle*config.classes_per_task
                    max_ = np.mod(min_ + config.classes_per_task, config.num_classes)
                    classes_list = np.arange(min_, max_) if max_ > min_ else np.concatenate(
                        [np.arange(min_, config.num_classes), np.arange(0, max_)])
                    if config.debug:
                        print(classes_list)
                    # wandb.log({"class_distribution_mean":mean})
                    wandb.log({"class_distribution_min": min_})
                    wandb.log({"class_distribution_max": max_})
                    classes = np.random.choice(classes_list, size=config.classes_per_task, replace=False)
                    task_id_in_cycle += 1
                elif config.class_sampling == 'cl_with_cycles':
                    wandb.log({"cycle_idx": cycle_idx})
                    cycle_size = int(config.num_classes / config.classes_per_task)
                    if max_ == config.num_classes:  # task_id % cycle_size==0 and task_id>0:
                        cycle_idx += 1
                        task_id_in_cycle = 0
                    wandb.log({"task_id_in_cycle": task_id_in_cycle})
                    min_ = task_id_in_cycle * config.classes_per_task
                    max_ = min_ + config.classes_per_task
                    classes = np.arange(min_, max_)
                    wandb.log({"class_distribution_min": min_})
                    wandb.log({"class_distribution_max": max_})
                    # classes = np.random.choice(classes_list, size=config.classes_per_task, replace=False)
                    task_id_in_cycle += 1
                else:
                    raise NotImplementedError

    return classes
