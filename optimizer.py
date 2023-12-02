import torch
import torch.optim


class Optimizers():
    def __init__(self, args):
        self.args = args


    def create(self, model, policy='no_bias_norm_decay'):
        if policy == 'regular':
            return torch.optim.SGD(model.parameters(), self.args.lr,
                                   momentum=self.args.momentum,
                                   weight_decay=self.args.weight_decay)
        elif policy == 'no_bias_norm_decay':
            params = self.no_bias_norm_decay(model)
            return torch.optim.SGD(params, self.args.lr, momentum=self.args.momentum)


    def no_bias_norm_decay_simple(self, model):
        # Separate parameters into groups
        weight_params = []
        bias_params = []
        bn_params = []

        for name, param in model.named_parameters():
            if 'weight' in name and ('conv' in name or 'fc' in name):
                weight_params.append(param)
            elif 'bias' in name:
                bias_params.append(param)
            elif 'bn' in name:
                bn_params.append(param)

        # No Bias/BN Decay: define separate parameter groups with specific regularization
        params = [
            {'params': weight_params, 'weight_decay': self.args.weight_decay},  # Apply weight decay to weights of conv and fc layers
            {'params': bias_params, 'weight_decay': 0},              # No weight decay for biases
            {'params': bn_params, 'weight_decay': 0},                # No weight decay for batch norm parameters
        ]

        return params


    # modified from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
    def no_bias_norm_decay(self, model):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Conv2d, torch.nn.Linear)
        blacklist_weight_modules = (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('alpha') or pn.endswith('beta'):
                    # Parameters ending in 'alpha' or 'beta' will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('in_proj_weight') or pn.endswith('pos_embedding') or pn.endswith('class_token'):
                    # Parameters ending in 'alpha' or 'beta' will not be decayed
                    no_decay.add(fpn)
                    
        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.args.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups