import torch
import torch.nn as nn
import transformers

from utils import recursive_getattr, recursive_setattr


class LoRALinear(torch.nn.Module):
    def __init__(self, weight, bias, lora_dim, lora_scaling):
        super(LoRALinear, self).__init__()
        # Save original weight and bias
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        # TODO: Implement lora left and right weights
        self.lora_dim = lora_dim
        if lora_dim > 0:
            self.lora_right_weight = nn.Parameter(torch.zeros(weight.shape[1], lora_dim))
            self.lora_left_weight  = nn.Parameter(torch.zeros(lora_dim, weight.shape[0]))
        else:
            # 如果 lora_dim=0, 说明不使用 LoRA
            self.lora_right_weight = None
            self.lora_left_weight  = None
        #############################################
        self.lora_scaling = lora_scaling / lora_dim if lora_dim > 0 else 0.0
        self.init_parameters()
        # TODO: Freeze original weight and bias
        #
        #######################################
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def init_parameters(self):
        # TODO: Initialize LoRA parameters
        ##################################
        if self.lora_dim > 0:
            # 官方 LoRA 论文中，通常将 lora_right_weight 用正常的 xavier 初始化，
            # 而将 lora_left_weight 用零初始化，也可以都做零初始化。
            nn.init.xavier_uniform_(self.lora_right_weight)
            nn.init.zeros_(self.lora_left_weight)

    def forward(self, input):
        # TODO: Implement the forward function
        ######################################
        out = input @ self.weight.transpose(0, 1)
        if self.bias is not None:
            out = out + self.bias

        # LoRA 分支
        if self.lora_dim > 0:
            lora_out = (input @ self.lora_right_weight) @ self.lora_left_weight
            out = out + self.lora_scaling * lora_out

        return out


def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1):
    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) or isinstance(module, transformers.pytorch_utils.Conv1D)) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        if isinstance(module, torch.nn.Linear):
            tmp = LoRALinear(module.weight, module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            tmp = LoRALinear(module.weight.t().detach(), module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        else:
            raise ValueError("Unsupported module type")
        recursive_setattr(model, name, tmp)
    return model


def only_optimize_lora_parameters(model):
    # TODO: Turn off the gradient of all the parameters except the LoRA parameters
    ##############################################################################
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            # 解冻 LoRA 参数
            if module.lora_right_weight is not None:
                module.lora_right_weight.requires_grad = True
            if module.lora_left_weight is not None:
                module.lora_left_weight.requires_grad = True

def get_lora_state_dict(model):
    # TODO: return lora left and right weights as state dict
    # The saved state dict will be used later for loading
    ########################################################
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.lora_dim > 0:
            lora_state_dict[name + ".lora_right_weight"] = module.lora_right_weight
            lora_state_dict[name + ".lora_left_weight"]  = module.lora_left_weight

    return lora_state_dict