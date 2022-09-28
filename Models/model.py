import torch
import torch.nn as nn          
import torch.nn.functional as F
import torchvision.models as models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from copy import deepcopy

class WeightNormLayer(nn.Module):
    def __init__(self, size_in, size_out, bias=True):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weight = nn.Parameter(torch.Tensor(size_out, size_in))
        if bias:
            self.bias = nn.Parameter(torch.zeros(size_out))
        else:
            self.bias = None

        # initialize weights
        nn.init.kaiming_normal_(self.weight)  # weight init

    def forward(self, x):
        x = F.linear(
            x, self.weight / torch.norm(self.weight, dim=1, keepdim=True), self.bias
        )
        return x

class Classifier(nn.Module):   
    def __init__(self, num_classes, in_d=100, head_name='linear', masking="single"):
        super().__init__()
        self.head_name=head_name
        self.masking = masking     
        self.num_classes = num_classes
        self.classes_mask = torch.eye(self.num_classes).to(device).float()
        if self.head_name == "weightnorm":
            self.head = WeightNormLayer(in_d, num_classes, bias=False)
        else:
            self.head = nn.Linear(in_d, num_classes)
    def forward(self,x:torch.Tensor):
        x=x.squeeze()  
        return self.head(x)
    
    def get_loss(self, out, labels, loss_func):
        if self.masking == "single":
            out = torch.mul(out, self.classes_mask[labels])
        elif self.masking == "group":
            label_unique = labels.unique()
            ind_mask = self.classes_mask[label_unique].sum(0)
            full_mask = ind_mask.unsqueeze(0).repeat(out.shape[0], 1)
            out = torch.mul(out, full_mask)
        loss = loss_func(out, labels.long())
        assert loss == loss, print("There should be some Nan")
        return loss

class Model_Wrapp(nn.Module):
    def __init__(
        self, model):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.head_name = head_name
        self.masking = masking
        self.input_dim = input_dim
        self.output_dim = 1
        self.image_size = 28
        self.features_size = 320
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(self.input_dim, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(self.features_size, 50)
        self.classes_mask = torch.eye(self.num_classes).to(device).float()
        self.hook_handle = None

        self.linear = nn.Sequential(
            self.fc1,
            self.relu,
        )  # for ogd
        if self.head_name == "weightnorm":
            self.head = WeightNormLayer(50, num_classes, bias=False)
        else:
            self.head = nn.Linear(50, num_classes)

    def feature_extractor(self, x):
        x = self.relu(self.maxpool2(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(-1, self.features_size)
        return self.linear(x)

    def forward(self, x, latent_vector=False):
        x = x.view(-1, self.input_dim, self.image_size, self.image_size)
        x = self.feature_extractor(x)
        if not latent_vector:
            x = self.head(x)
        return x

    def get_loss(self, out, labels, loss_func):
        if self.masking == "single":
            out = torch.mul(out, self.classes_mask[labels])
        elif self.masking == "group":
            label_unique = labels.unique()
            ind_mask = self.classes_mask[label_unique].sum(0)
            full_mask = ind_mask.unsqueeze(0).repeat(out.shape[0], 1)
            out = torch.mul(out, full_mask)
        loss = loss_func(out, labels.long())
        assert loss == loss, print("There should be some Nan")
        return loss

class Model(nn.Module):
    def __init__(
        self, num_classes=10, input_dim=1, head_name="linear", masking="single"
    ):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.head_name = head_name
        self.masking = masking
        self.input_dim = input_dim
        self.output_dim = 1
        self.image_size = 28
        self.features_size = 320
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(self.input_dim, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(self.features_size, 50)
        self.classes_mask = torch.eye(self.num_classes).to(device).float()
        self.hook_handle = None
        self.conv1_save = None

        self.linear = nn.Sequential(
            self.fc1,
            self.relu,
        )  # for ogd
        if self.head_name == "weightnorm":
            self.head = WeightNormLayer(50, num_classes, bias=False)
        else:
            self.head = nn.Linear(50, num_classes)

    def feature_extractor(self, x):
        x = self.relu(self.maxpool2(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(-1, self.features_size)
        return self.linear(x)

    def forward(self, x, latent_vector=False):
        x = x.view(-1, self.input_dim, self.image_size, self.image_size)
        x = self.feature_extractor(x)
        if not latent_vector:
            x = self.head(x)
        return x

    def track_weight_diff(self):
        diff_1, diff_2, diff_3, diff_4, diff_5 = None, None, None, None, None
        if self.conv1_save is not None:
            diff_1 = (self.conv1_save - self.conv1.weight.detach()).mean().item()
            diff_2 = (self.conv2_save - self.conv2.weight.detach()).mean().item()
            diff_3 = (self.fc1_save - self.fc1.weight.detach()).mean().item()
            diff_4 = (self.fc2_save - self.linear[0].weight.detach()).mean().item()
            diff_5 = (self.fc3_save - self.head.weight.detach()).mean().item()

        self.conv1_save = deepcopy(self.conv1.weight.detach())
        self.conv2_save = deepcopy(self.conv2.weight.detach())
        self.fc1_save = deepcopy(self.fc1.weight.detach())
        self.fc2_save = deepcopy(self.linear[0].weight.detach())
        self.fc3_save = deepcopy(self.head.weight.detach())
        
        return diff_1, diff_2, diff_3, diff_4, diff_5

    def track_grads(self):
        grad_1 = self.conv1.weight.grad.mean().item()
        grad_2 = self.conv2.weight.grad.mean().item()
        grad_3 = self.fc1.weight.grad.mean().item()
        grad_4 = self.linear[0].weight.grad.mean().item()
        grad_5 = self.head.weight.grad.mean().item()
        return grad_1, grad_2, grad_3, grad_4, grad_5

class EncoderClassifier(nn.Module):                 
    def __init__(self, encoder:nn.Module, classifier:Classifier):
        super().__init__()
        self.encoder = encoder      
        self.classifier = classifier
    def forward(self,x, *args, **kwargs):
        return self.classifier(self.encoder(x,*args,**kwargs))
    def get_loss(self, out, labels, loss_func):
        return self.classifier.get_loss(out,labels,loss_func)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


from Models.resnet import cifar_resnet20, cifar_resnet32, cifar_resnet44, cifar_resnet56

def get_CIFAR_Model(
    config,
    num_classes=10,
    head_name="linear",
    masking="single",
    nb_layers=20,
    pretrained_on=None,
    model_dir="./Models",
    finetuning=False,
):

    if config.architecture=="default":
        model = CIFARModel(num_classes, head_name, masking, nb_layers, pretrained_on, model_dir,finetuning)
    elif config.architecture == "resnet":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif config.architecture == 'vgg':
        model = models.vgg16()
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif config.architecture == 'vit_b_16':
        model = models.vit_b_16()
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif config.architecture == 'inception':
        model = models.inception_v3(aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif config.architecture == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    return model

def CIFARModel(num_classes=10,
            head_name="linear",
            masking="single",
            nb_layers=20,
            pretrained_on=None,
            model_dir="./Models",
            finetuning=False):

    if pretrained_on is not None:
        if nb_layers == 20:
            model = cifar_resnet20(
                num_classes,
                pretrained=pretrained_on,
                model_dir=model_dir,
                head_name=head_name,
                masking=masking,
            )
    else:

        if nb_layers == 20:
            model = cifar_resnet20(num_classes, head_name=head_name, masking=masking)
        elif nb_layers == 32:
            model = cifar_resnet32(num_classes, head_name=head_name, masking=masking)
        elif nb_layers == 44:
            model = cifar_resnet44(num_classes, head_name=head_name, masking=masking)
        elif nb_layers == 56:
            model = cifar_resnet56(num_classes, head_name=head_name, masking=masking)

    if (pretrained_on is not None) and (not finetuning):
        model = freeze_model(model)

    return model
