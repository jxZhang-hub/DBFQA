import torch
import numpy as np
from scipy import stats
import random
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import tqdm
import gc
import logging
import argparse

from models.DBFQA import Net
from dataset.Dataset import IQADataset

def get_index(num):
    random.seed(1)
    index = list(range(num))
    random.shuffle(index)
    return index

def set_logging():
    if not os.path.exists("./logs/"):
        os.makedirs("./logs/")
    filename = "./logs/record.log"
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DB-FQA training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_count', type=int, default=10000)
    parser.add_argument('--train_rate', type=float, default=0.8)

    args = parser.parse_args()
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    data_count = args.data_count
    train_rate = args.train_rate
    num_workers = args.num_workers


    set_logging()
    cuda = torch.cuda.is_available()
    index = get_index(data_count)
    train_idx = index[:int(train_rate * len(index))]
    test_idx = index[int((train_rate) * len(index)):]
    train_set = IQADataset(train_idx)
    test_set = IQADataset(test_idx)

    print("train set size: "+str(train_set.__len__()))
    print("test set size : "+ str(test_set.__len__()))

    device = torch.device('cuda' if cuda else 'cpu')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True)


    model = Net(batch_size=batch_size).to(device)

    logging.info("Start")
    #ckpt_path = ""
    #model.load_state_dict(torch.load(ckpt_path, weights_only=True), strict=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)

    best_srcc = 0
    for epoch in range(epochs):
        # train
        model.train()
        LOSS = 0
        with tqdm.tqdm(total=train_set.__len__()//batch_size) as pbar:
            for i, (image, label) in enumerate(train_loader):
                imgg = image[0].to(device)
                imge = image[1].to(device)
                label = label.to(device)
                optimizer.zero_grad()

                outputs = model(imgg, imge)

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                scheduler.step()
                LOSS = LOSS + loss.item()
                pbar.update(1)
                gc.collect()
                torch.cuda.empty_cache()
        print("train loss: ", LOSS/(len(train_set) // batch_size))
        train_loss = LOSS / (i + 1)
        # val

        if epoch % 1 == 0:
            gt = []
            y_pred = []
            model.eval()
            with torch.no_grad():
                with tqdm.tqdm(total=test_set.__len__()//batch_size) as pbar:
                    for i, (image, label) in enumerate(test_loader):
                        imgg = image[0].to(device)
                        imge = image[1].to(device)
                        label = label.to(device)
                        outputs = model(imgg, imge)

                        label = label.cpu().numpy()
                        outputs = outputs.cpu().numpy()

                        gt = np.append(gt, label)
                        y_pred = np.append(y_pred, outputs)
                        pbar.update(1)

            val_SROCC = stats.spearmanr(y_pred, gt)[0]
            val_PLCC = stats.pearsonr(y_pred, gt)[0]
            val_KROCC = stats.stats.kendalltau(y_pred, gt)[0]
            val_RMSE = np.sqrt(((y_pred - gt) ** 2).mean())

            print("Epoch " + str(epoch+1)+"\tSROCC:"+str(val_SROCC),"PLCC:"+str(val_PLCC),"KROCC:"+str(val_KROCC),"RMSE:"+str(val_RMSE))
            logging.info('epoch{}, SRCC:{}, PLCC:{}, KRCC:{}, RMSE:{}'.format(epoch + 1, val_SROCC, val_PLCC, val_KROCC, val_RMSE))
            if val_SROCC > best_srcc:
                best_srcc = val_SROCC
                torch.save(model.state_dict(), './models/checkpoints/model_{:.5f}.pth'.format(best_srcc))
                print('save model')