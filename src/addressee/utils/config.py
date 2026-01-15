import yaml
from types import SimpleNamespace



def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)



def deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if (
            k in base
            and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(args, extra_args):
    cfg = load_yaml("src/addressee/models/"+args.model_type+"/config.yml")
    if args.exp_config is not None: 
        cfg_exp = load_yaml("src/addressee/config/" + args.exp_config)
        #merge default config with experience config
        cfg = deep_merge(cfg, cfg_exp)
    # merge updated config with run args
    cfg = deep_merge(cfg, vars(args))
    cfg = deep_merge(cfg, vars(parse_unknown_args(extra_args)))
    #cfg = SimpleNamespace(**cfg)
    cfg = to_namespace(cfg)
    return cfg

def save_config(cfg, save_path):
    with open(save_path / "config.yaml", "w") as f:
        yaml.dump(cfg, f)


def to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [to_namespace(x) for x in obj]
    else:
        return obj
    
def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {
            k: namespace_to_dict(v)
            for k, v in vars(obj).items()
        }
    elif isinstance(obj, list):
        return [namespace_to_dict(x) for x in obj]
    else:
        return obj


def parse_unknown_args(unknown):
    ns = SimpleNamespace()
    it = iter(unknown)

    for token in it:
        if token.startswith("--"):
            key = token[2:].replace("-", "_")

            try:
                value = next(it)
                if value.startswith("--"):
                    setattr(ns, key, True)
                    it = iter([value] + list(it))
                else:
                    setattr(ns, key, value)
            except StopIteration:
                setattr(ns, key, True)

    return ns