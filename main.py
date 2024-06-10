import torch
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from torch.utils.data import DataLoader

from eval import Eval
from models.moduleRCA import *
from data.load_data import get_path
from data.dataset import CustomDataset
from Options.TrainOptions import TrainOption
from Options.TestOptions import TestOption
from train.trainer import Trainer
from utils.seed import seed_everything
from utils.loss import BCEDiceLoss

def main(train_args):
    device = train_args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    seed_everything(train_args)

    # get path
    get_path(train_args)

    train_ct = pd.read_excel(f"D:\\HIPPO\\train_.xlsx")['CT']
    train_hippo = pd.read_excel(f"D:\\HIPPO\\train_.xlsx")['HIPPO']

    test_ct = pd.read_excel(f"D:\\HIPPO\\test_.xlsx")['CT']
    test_hippo = pd.read_excel(f"D:\\HIPPO\\test_.xlsx")['HIPPO']

    print(f"Load Data\nTrain : {len(train_ct)} Test : {len(test_ct)}")

    # define models
    # Unet, Unet edge, Unet module
    model = Model(1, 1).to(device)

    model = torch.nn.DataParallel(model).to(device)
    print(f"Model Parameter : {sum(p.numel() for p in model.parameters())}")
    # trainer
    trainer = Trainer(train_args,
                      train_ct,
                      train_hippo,
                      model)
    
    trainer.train()

    # evaluation
    # evaluator = Eval(test_args,
    #                  test_ct,
    #                  test_hippo,
    #                  model)
    
    # evaluator.evaluation()

def get_parser(Parser):
    parser = Parser()
    return parser.parse()

if __name__=='__main__':
    train_args = get_parser(TrainOption)
    print(train_args)
    # test_args = get_parser(TestParser) 
    # print(test_args)
    main(train_args)