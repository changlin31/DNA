import os
import yaml
import argparse


class Initial:
    def __init__(self, args, base_configs=None, hyperparam_config=None):
        # parser = argparse.ArgumentParser(description='AutoTrainInit')
        self.args = args
        if base_configs:
            for base_config in base_configs:
                self.base_init = self.parser('initialize/'+base_config)
        if hyperparam_config:
            self.hyper_param_config = self.parser(os.path.join(args.cache_root, hyperparam_config))

    def parser(self, config_yaml):
        with open(config_yaml, 'r') as f:
            config = yaml.safe_load(f)
        arg_dict = self.args.__dict__
        for key, value in config.items():
            arg_dict[key] = value
        return config
