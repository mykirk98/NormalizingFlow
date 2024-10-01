import argparse
import os
import torch
from ignite.contrib import metrics
import constants as const
import dataset
import NF_model as NF_model
import utils
import time
import torch.nn as nn
import json
from collections import OrderedDict
import cv2
import numpy as np 


best_AUROC = 0


def save_output(data, target, outputs, index, size = (512,512)):
    def convert_img_to_np(img, size):
        img= img.cpu().detach().numpy()
        img = np.moveaxis(a=img, source=0, destination=-1) * 255
        if img.shape[2] == 1:
            # cv2.imshow('output', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            k = cv2.cvtColor(src=img, code=cv2.COLOR_GRAY2BGR)
            return  k.astype(np.uint8)
        else: return img
    data = convert_img_to_np(img=data[index], size=size)
    outputs = convert_img_to_np(img=outputs[index], size=size)
    target = convert_img_to_np(img=target[index], size=size)
    cv2.imwrite(filename='data.png', img=data)
    cv2.imwrite(filename='target.png', img=target)
    cv2.imwrite(filename='output.png', img=outputs)



def build_train_data_loader(args):
    # train_dataset = dataset.MoldDataset(root=args.data, input_size=256, is_train=True)
    train_dataset = dataset.MVTecDataset(root=args.data, category=args.category, input_size=256, is_train=True)
    
    # return torch.utils.data.DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    return torch.utils.data.DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, drop_last=True)



def build_test_data_loader(args):
    # test_dataset = dataset.MoldDataset(root=args.data, input_size=256, is_train=False)
    test_dataset = dataset.MVTecDataset(root=args.data, category=args.category, input_size=256, is_train=False)

    # return torch.utils.data.DataLoader(test_dataset, batch_size=const.BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)
    return torch.utils.data.DataLoader(test_dataset, batch_size=const.BATCH_SIZE, shuffle=False, drop_last=False)



def norm_inv(image):
    image = (image+ (np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225]))) * np.array([0.229, 0.224, 0.225])

    return (image*255).astype(np.uint8)



def build_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NF_model.Model()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(module=model.to(device=device))
    else:
        model = model.to(device=device)

    print(f"Model A.D. Param#: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model



def build_optimizer(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(module=model.to(device=device))
    else:
        model = model.to(device=device)

    return torch.optim.Adam(model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY)



def eval_once(dataloader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(module=model.to(device=device))
    else:
        model = model.to(device=device)

    model.eval()
    auroc_metric = metrics.ROC_AUC()
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    avg_time = 0
    save_batch  =   True
    for i, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        t0 = time.time()
        with torch.no_grad():
            ret = model(data)
        avg_time += time.time() - t0
        outputs = ret["anomaly_map"].cpu().detach()
        
        if save_batch:
            anom_maps   =   torch.nn.functional.sigmoid(outputs).numpy()
            for index_map, anom_map in enumerate(anom_maps):
                img_orig    =   norm_inv(data[index_map,:,:,:].permute(1,2,0).detach().cpu().numpy())
                target_img  =   ((targets[index_map,:,:,:].permute(1,2,0).detach().cpu().numpy()[:,:,0])*255).astype(np.uint8)
                anom_mask   =   (anom_map[0,:,:]*255).astype(np.uint8)
                stack_masks =   cv2.cvtColor(np.hstack((target_img, anom_mask )),cv2.COLOR_GRAY2BGR)
                stack_in_mas =  np.hstack((img_orig,stack_masks))
                cv2.imwrite(f'anom_maps/test_{index_map}.png',stack_in_mas)
                #save_batch  =   False
                
        save = True
        if i == 1 and save:
            save_output(data, targets, outputs, 4)

        outputs = outputs.flatten()
        targets = targets.flatten()
        targets[targets > 0] = 1
        auroc_metric.update((outputs, targets))
    auroc = auroc_metric.compute()
    print(f"AUROC: {auroc}")
    print(f"Avg time: {avg_time / len(dataloader)}, Data num: {len(dataloader)}")



def eval_once_on_training(dataloader, model, checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(module=model.to(device=device))
    else:
        model = model.to(device=device)

    model.eval()
    auroc_metric = metrics.ROC_AUC()
    avg_time = 0

    scores_list = []
    labels_list = []
    save_batch = True
    for data, targets in dataloader:
        data, targets = data.to(device), targets
        t0 = time.time()
        with torch.no_grad():
            ret = model(data)
            if save_batch:
                anom_maps   =   torch.nn.functional.sigmoid(ret['anomaly_map']).detach().cpu().numpy()
            for index_map, anom_map in enumerate(anom_maps):
                cv2.imwrite(f'anom_maps/{index_map}.png',(anom_map[0,:,:]*255).astype(np.uint8))
                save_batch  =   False
        avg_time += time.time() - t0
        outputs = ret["anomaly_map"].cpu().detach()

        outputs = outputs.flatten()
        targets = targets.flatten()
        targets[targets > 0] = 1

        scores_list.append(outputs)
        labels_list.append(targets)

    scores = torch.cat(scores_list)
    labels = torch.cat(labels_list)
    auroc_metric.update((scores, labels))

    auroc = auroc_metric.compute()
    print(f"AUROC: {auroc}")

    global best_AUROC
    if auroc > best_AUROC:
        best_AUROC = auroc
        torch.save(model.state_dict(), os.path.join(checkpoint_dir,"best.pt"))
        print(f"Best AUROC: {best_AUROC}")

def train_one_epoch(dataloader, model, optimizer, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(module=model.to(device=device))
    else:
        model = model.to(device=device)
    
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.to(device)
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )



def train(args):
    os.makedirs(name=os.path.join(const.CHECKPOINT_DIR, args.category), exist_ok=True)
    os.makedirs(name=const.ANOM_MAPS_LOCATION, exist_ok=True)
    checkpoint_dir = os.path.join(const.CHECKPOINT_DIR, args.category)
    # checkpoint_dir_log = os.path.join(
    #     const.CHECKPOINT_DIR, "log%d" % len(os.listdir(const.CHECKPOINT_DIR))
    # )
    os.makedirs(name=checkpoint_dir, exist_ok=True)
    # os.makedirs(checkpoint_dir_log, exist_ok=True)

    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(module=model.to(device=device))
    else:
        model = model.to(device=device)

    optimizer = build_optimizer(model=model)

    train_dataloader = build_train_data_loader(args=args)
    test_dataloader = build_test_data_loader(args=args)

    for x in train_dataloader:
        print(x.size())
    print(f"Train data num: {train_dataloader.__len__()*32}")
    print(f"Test data num: {test_dataloader.__len__()*32}")
    global best_AUROC
    best_AUROC = 0
    
    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(dataloader=train_dataloader, model=model, optimizer=optimizer, epoch=epoch)

        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once_on_training(dataloader=test_dataloader, model=model, checkpoint_dir=checkpoint_dir)
    if epoch==(const.NUM_EPOCHS-1):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
             os.path.join(checkpoint_dir, "best.pt")
        )

    print(f"Best AUROC: {best_AUROC}")

    with open(file="result.txt", mode="a") as f:
        f.write(f"{best_AUROC}\n")

def evaluate(args):
    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(module=model.to(device=device))
    else:
        model = model.to(device=device)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_dataloader = build_test_data_loader(args)
    print(f"Test data num: {test_dataloader.__len__()*32}")
    eval_once(test_dataloader, model)

def parse_args():
    parser = argparse.ArgumentParser(description="Train on MVTec-AD dataset")
    parser.add_argument("--data", dest='data', type=str, required=True, help="path to mvtec folder")
    # parser.add_argument("-cat", "--category", type=str, choices=const.MVTEC_CATEGORIES, required=True, help="category name in mvtec")
    parser.add_argument("-cat", "--category", dest='category', type=str, help="category name in mvtec")
    parser.add_argument("--eval", dest='eval', action="store_true", help="run eval only")
    parser.add_argument("-ckpt", "--checkpoint", dest='checkpoint', type=str, help="path to load checkpoint")
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    args = parse_args()
    if args.eval:
        # python main.py --data path/to/data/ --category bottle --eval --check_points path/to/pt_file
        start = time.time()
        evaluate(args)
        end = time.time()
        print(f"Total time: {end - start}")
    else:
        # python main.py --data path/to/data/ --category bottle
        start = time.time()
        train(args)
        end = time.time()
        print(f"Total time: {end - start}")