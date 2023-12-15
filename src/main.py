import hydra
from omegaconf import DictConfig
from model import RGT
#from data_preparation import somethinggg

@hydra.main(version_base=None, config_path="config", config_name="RGT")
def main(cfg: DictConfig):
    print(cfg)
    #rgt = RGT(cfg)

if __name__ == '__main__':
    main()
