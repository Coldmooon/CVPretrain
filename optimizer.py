import torch
import torch.optim


class Optimizers():
    def __init__(self, args):
        self.args = args


    def create(self, model, optim='sgd', policy='no_bias_norm_decay'):
        if policy == 'regular':
            params = model.parameters()
            global_weight_decay = self.args.weight_decay
        elif policy == 'no_bias_norm_decay':
            params = self.no_bias_norm_decay(model)
            global_weight_decay = 0 # optim_groups will take precedence over the global weight_decay setting


        if optim == 'sgd':
            optimizer = torch.optim.SGD(params, 
                                   self.args.lr,
                                   momentum=self.args.momentum,
                                   weight_decay=global_weight_decay)
        elif optim == 'adam':
            optimizer = torch.optim.Adam(params, 
                                    self.args.lr, 
                                    betas=(0.9, 0.999), 
                                    eps=1e-08, 
                                    weight_decay=global_weight_decay)
        elif optim == 'adamw':
            optimizer = torch.optim.AdamW(params, 
                                          self.args.lr,
                                          weight_decay=global_weight_decay)
        else:
            raise ValueError('optimizer not supported: {}'.format(optim))


        for i, group in enumerate(optimizer.param_groups):
            print(f"Parameter group {i} weight_decay: {group['weight_decay']}")
        
        return optimizer

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
        # Note: Different implements of transformer block usually have different name style. So, for Vit 
        # or other transformer models, you should print(pn) to check which layer.weights or bias should be
        # decay.
        whitelist_weight_modules = (torch.nn.Conv2d,
                                    torch.nn.Linear)
        
        blacklist_weight_modules = (torch.nn.BatchNorm2d,
                                    torch.nn.LayerNorm,
                                    torch.nn.GroupNorm,
                                    torch.nn.InstanceNorm2d,
                                    torch.nn.LocalResponseNorm,
                                    torch.nn.Embedding)
        
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
                elif pn.endswith('in_proj_weight'):
                    decay.add(fpn)
                # for Transformer block:
                elif pn.endswith('pos_embedding') or pn.endswith('token') or pn.endswith('relative_position_bias_table'):
                    no_decay.add(fpn)
                    
        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        # param_dict is constructed by {parameter_name: parameter}
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        gradient_check = False
        if not gradient_check:
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.args.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
        else:
            optim_groups = []
            for pn in sorted(list(decay)):
                optim_groups.append({"params": param_dict[pn], "weight_decay": self.args.weight_decay, "layer_name": pn})
            for pn in sorted(list(no_decay)):
                optim_groups.append({"params": param_dict[pn], "weight_decay": 0.0, "layer_name": pn})

        return optim_groups
