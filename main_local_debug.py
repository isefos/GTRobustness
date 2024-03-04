from main import ex
import yaml
import torch


torch.autograd.set_detect_anomaly(True, check_nan=True)



with open("configs_sacred/test_cora.yaml") as f:
    cfg = yaml.safe_load(f)

r = ex.run(config_updates=cfg)
