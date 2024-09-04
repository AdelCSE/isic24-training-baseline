import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import WeightedRandomSampler

sys.path.append('../..')
from src.datasets import ISICDataset, ISICTrainDataset
from src.models import CoATNet, EVA02, EffNet, ConvNext
from src.trainer import Trainer
from src.utils import load_env_vars, history2df


class DEFAULT:
    SEED = 42
    NUM_EPOCHS = 5
    IN_CHANNELS = 3
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 64
    NEG_SAMPLES_MULTIPLIER = 20
    LEARNING_RATE = 1e-5
    NUM_ACCUMULATION = 1
    WEIGHT_DECAY = 1e-6
    MIN_LR = 1e-6
    NUM_FOLDS = 5
    FOLD = 0
    T_MAX = 500
    SEED = 42
    SCHEDULER = 'CosineAnnealingLR'


class EVA02_PARAMS:
    MODEL_NAME = 'eva02_small_patch14_336.mim_in22k_ft_in1k'
    IMG_SIZE = 336

class COATNET_PARAMS:
    MODEL_NAME = 'maxvit_tiny_tf_224.in1k'
    IMG_SIZE = 224

class EFFNET_PARAMS:
    MODEL_NAME = 'tf_efficientnet_b0_ns'
    IMG_SIZE = 384

class CONVNEXT_PARAMS:
    MODEL_NAME = 'convnext_tiny.fb_in22k_ft_in1k_384'
    IMG_SIZE = 384
    
class GLOBAL:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 1


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_loaders(args, df_train, df_valid, hdf5_file, data_transforms):        

    train_dataset = ISICTrainDataset(df=df_train, hdf5_file=hdf5_file, transforms=data_transforms['train'])

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=DEFAULT.TRAIN_BATCH_SIZE, 
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True, 
                              num_workers=args.num_workers,
                              prefetch_factor=args.prefetch_factor)
    
    if df_valid is not None:
        valid_dataset = ISICDataset(df=df_valid, hdf5_file=hdf5_file, transforms=data_transforms['valid'])
        valid_loader = DataLoader(dataset=valid_dataset, 
                                  batch_size=DEFAULT.VALID_BATCH_SIZE, 
                                  shuffle=False, 
                                  pin_memory=True, 
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch_factor)
        
        return train_loader, valid_loader
    
    return train_loader, None

def create_transforms(image_size: int):
    data_transforms = {
        'train': A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()
        ], p=1.0),

        'valid': A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()
        ], p=1.0)
    }

    return data_transforms


def create_model(args) -> tuple[nn.Module, nn.Module]:

    model, criterion = None, None

    if args.model == 'eva02':

        model = EVA02(model_name=EVA02_PARAMS.MODEL_NAME,
                          in_channels=DEFAULT.IN_CHANNELS,
                          num_classes=GLOBAL.NUM_CLASSES,
                          pretrained=True
                          ).to(GLOBAL.DEVICE)
            
    elif args.model == 'coatnet':
        model = CoATNet(model_name=COATNET_PARAMS.MODEL_NAME,
                            in_channels=DEFAULT.IN_CHANNELS,
                            num_classes=GLOBAL.NUM_CLASSES,
                            pretrained=True
                            ).to(GLOBAL.DEVICE)
    
    elif args.model == 'convnext':
        model = ConvNext(model_name=CONVNEXT_PARAMS.MODEL_NAME,
                            in_channels=DEFAULT.IN_CHANNELS,
                            num_classes=GLOBAL.NUM_CLASSES,
                            pretrained=True
                            ).to(GLOBAL.DEVICE)
    
    elif args.model == 'effnet':
        model = EffNet(model_name=EFFNET_PARAMS.MODEL_NAME,
                            in_channels=DEFAULT.IN_CHANNELS,
                            num_classes=GLOBAL.NUM_CLASSES,
                            pretrained=True
                            ).to(GLOBAL.DEVICE)

    else:
        raise Exception(f'Model Not Found Error: {args.model} is not a valid model name.')
    
    criterion = nn.BCELoss()

    return model, criterion


def get_scheduler(optimizer, t_max):
    
    if DEFAULT.SCHEDULER == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=t_max, eta_min=DEFAULT.MIN_LR)
        print('Using CosineAnnealingLR Scheduler!\n')

    elif DEFAULT.SCHEDULER == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=t_max, T_mult=1, eta_min=DEFAULT.MIN_LR)
        print('Using CosineAnnealingWarmRestarts Scheduler!\n')

    else:
        return None
    
    return scheduler

def downsample_data(df, multiplier):

    df_positive = df[df["target"] == 1].reset_index(drop=True)
    df_negative = df[df["target"] == 0].reset_index(drop=True)

    df = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*multiplier, :]]).reset_index(drop=True)

    return df


def main(args):

    seed_everything(DEFAULT.SEED)

    print(f'Starting training with the following args: \n {args}')
    print(f'Using {GLOBAL.DEVICE} device for training!\n')

    DATA_DIR, MODELS_DIR, HISTORIES_DIR = load_env_vars()
    
    if (args.data_folder is not None):
        data_dir = os.path.join(DATA_DIR, args.data_folder)
    else:
        data_dir = DATA_DIR
        
    models_dir = os.path.join(MODELS_DIR, args.weights_folder)
    histories_dir = os.path.join(HISTORIES_DIR, args.histories_folder)

    if not os.path.exists(data_dir):
        raise Exception(f'Data directory not found: {data_dir}')
    
    if not os.path.exists(models_dir):
        raise Exception(f'Models directory not found: {models_dir}')
    
    if not os.path.exists(histories_dir):
        raise Exception(f'Histories directory not found: {histories_dir}')
    
    # Load data
    df = pd.read_csv(os.path.join(data_dir, 'train-metadata.csv'), low_memory=False)

    df = downsample_data(df, multiplier=20)

    # Get data transforms
    if args.model == 'coatnet':
        data_transforms = create_transforms(image_size=COATNET_PARAMS.IMG_SIZE)
    elif args.model == 'effnet':
        data_transforms = create_transforms(image_size=EFFNET_PARAMS.IMG_SIZE)
    elif args.model == 'convnext':
        data_transforms = create_transforms(image_size=CONVNEXT_PARAMS.IMG_SIZE)
    else:
        data_transforms = create_transforms(image_size=EVA02_PARAMS.IMG_SIZE)
    hdf5_file_path = os.path.join(data_dir, 'train-image.hdf5')

    if(args.mode == 'cv'):
        # Straified Group KFold

        sgkf = StratifiedGroupKFold(n_splits=DEFAULT.NUM_FOLDS)
        for fold, (train_idx, valid_idx) in enumerate(sgkf.split(X=df, y=df.target, groups=df.patient_id)):
    
            print(f'\nTraining Fold: {fold + 1} \n')

            df_train = df.loc[train_idx].reset_index(drop=True)
            df_valid = df.loc[valid_idx].reset_index(drop=True)

            print(f'Training with {df_train.shape[0]} samples: {df_train[df_train.target == 1].shape[0]} Positives and {df_train[df_train.target == 0].shape[0]} Negatives\n')
                
            # Get data loaders
            train_loader, valid_loader = create_loaders(args, df_train=df_train, df_valid=df_valid, hdf5_file=hdf5_file_path, data_transforms=data_transforms)
            
            # Get model and loss
            model, criterion = create_model(args)
            
            # Define optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT.LEARNING_RATE, weight_decay=DEFAULT.WEIGHT_DECAY)
            
            # Define scheduler
            t_max =  df.shape[0] * (DEFAULT.NUM_FOLDS - 1) * DEFAULT.NUM_EPOCHS // DEFAULT.TRAIN_BATCH_SIZE // DEFAULT.NUM_FOLDS
            scheduler = get_scheduler(optimizer, t_max)
            
            # Define trainer
            trainer = Trainer(weights_folder=models_dir) \
                .set_optimizer(optimizer=optimizer) \
                .set_criterion(criterion=criterion) \
                .set_scheduler(scheduler=scheduler) \
                .set_device(device=GLOBAL.DEVICE)
            
            # Start training
            trainer.train(model=model, train_loader=train_loader, val_loader=valid_loader, num_accum=DEFAULT.NUM_ACCUMULATION, fold=fold, num_epochs=args.epochs)
            
            # Save history
            history2df(trainer.history).to_csv(os.path.join(histories_dir, f'fold_{fold}.csv'), index=False)

    else:
        # Train on whole dataset
        print(f'\n Training on Whole Dataset \n')

        # Get train loader
        train_loader, _ = create_loaders(args, df=df, fold=None, hdf5_file=hdf5_file_path, data_transforms=data_transforms)
    
        # Get model and loss
        model, criterion = create_model(args)
    
        # Define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT.LEARNING_RATE, weight_decay=DEFAULT.WEIGHT_DECAY)
    
        # Define scheduler
        t_max =  df.shape[0] * DEFAULT.NUM_EPOCHS // DEFAULT.TRAIN_BATCH_SIZE
        scheduler = get_scheduler(optimizer, t_max)
    
        # Define trainer
        trainer = Trainer(weights_folder=models_dir) \
            .set_optimizer(optimizer=optimizer) \
            .set_criterion(criterion=criterion) \
            .set_scheduler(scheduler=scheduler) \
            .set_device(device=GLOBAL.DEVICE)
        
        # Start training
        trainer.train(model=model, train_loader=train_loader, val_loader=None, num_accum=DEFAULT.NUM_ACCUMULATION, fold=None, num_epochs=args.epochs)

    print('Training Done!')

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    ### General Arguments
    parser.add_argument("--model", type=str, choices=['eva02', 'coatnet', 'effnet', 'convnext'], required=True, help="Model name to train!")
    parser.add_argument("--data-folder", type=str, choices=['isic_2024'], required=True, help="Data directory name!")
    parser.add_argument("--weights-folder", type=str, required=True, help="Folder to save model weights!")
    parser.add_argument("--histories-folder", type=str, required=True, help="Folder to save training histories!")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train the model!")
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loaders!')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='Data loader prefetch factor!')
    parser.add_argument('--mode', type=str, choices=['train', 'cv'], required=True, help='Training whole dataset or using cross-validation!')

    args = parser.parse_args()

    main(args)