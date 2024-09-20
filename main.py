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
    import cv2
    import numpy as np 
    def convert_img_to_np(img, size):
        img= img.cpu().detach().numpy()
        img = np.moveaxis(img, 0, -1)*255
        if img.shape[2] == 1:
            # cv2.imshow('output', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            k = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return  k.astype(np.uint8)
        else: return img
    data = convert_img_to_np(data[index], size = size)
    outputs = convert_img_to_np(outputs[index], size = size)
    target = convert_img_to_np(target[index], size = size)
    cv2.imwrite('data.png',data)
    cv2.imwrite('target.png',target)
    cv2.imwrite('output.png',outputs)


def build_train_data_loader(args):
    train_dataset = dataset.MoldDataset(
        root=args.data,
        # category=args.category,
        input_size=256,
        is_train=True,
    )
    # train_dataset = dataset.MoldDataset(root=args.data,# category=args.category,
    #     input_size=256,
    #     is_train=True,
    # )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        #num_workers=4,
        drop_last=True,
    )

def norm_inv(image):
    image = (image+ (np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225]))) * np.array([0.229, 0.224, 0.225])
    return (image*255).astype(np.uint8)

def build_test_data_loader(args):
    test_dataset = dataset.MoldDataset(
        root=args.data,
        # category=args.category,
        input_size=256,
        is_train=False,
    )
    # dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        #num_workers=4,
        drop_last=False,
    )

def build_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NF_model.Model()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(module=model.to(device=device))
    else:
        model = model.to(device=device)
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
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
    print("AUROC: {}".format(auroc))
    print("Avg time: {}, Data num: {}".format(avg_time / len(dataloader), len(dataloader)))

def eval_once_on_trainig(dataloader, model, checkpoint_dir):
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
    print("AUROC: {}".format(auroc))

    global best_AUROC
    if auroc > best_AUROC:
        best_AUROC = auroc
        torch.save(model.state_dict(), os.path.join(
                                            checkpoint_dir,
                                            "best.pt"))
        print("Best AUROC: {}".format(best_AUROC))

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
    os.makedirs(os.path.join(const.CHECKPOINT_DIR, args.category), exist_ok=True)
    os.makedirs(const.ANOM_MAPS_LOCATION, exist_ok=True)
    checkpoint_dir = os.path.join(const.CHECKPOINT_DIR, args.category)
    # checkpoint_dir_log = os.path.join(
    #     const.CHECKPOINT_DIR, "log%d" % len(os.listdir(const.CHECKPOINT_DIR))
    # )
    os.makedirs(checkpoint_dir, exist_ok=True)
    # os.makedirs(checkpoint_dir_log, exist_ok=True)

    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(module=model.to(device=device))
    else:
        model = model.to(device=device)

    optimizer = build_optimizer(model=model)

    train_dataloader = build_train_data_loader(args)
    test_dataloader = build_test_data_loader(args)
    for x in train_dataloader:
        print(x.size())
    print("Train data num: {}".format(train_dataloader.__len__()*32))
    print("Test data num: {}".format(test_dataloader.__len__()*32))
    global best_AUROC
    best_AUROC = 0
    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)

        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once_on_trainig(test_dataloader, model, checkpoint_dir)
    if epoch==(const.NUM_EPOCHS-1):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
             os.path.join(checkpoint_dir, "best.pt"),
        )
    print("Best AUROC: {}".format(best_AUROC))
    txt_file = "result.txt"
    with open(txt_file, "a") as f:
        f.write("{}\n".format(best_AUROC))

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
    print("Test data num: {}".format(test_dataloader.__len__()*32))
    eval_once(test_dataloader, model)

def parse_args():
    parser = argparse.ArgumentParser(description="Train on MVTec-AD dataset")
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument("-cat", "--category", type=str, choices=const.MVTEC_CATEGORIES, required=True, help="category name in mvtec")
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument("-ckpt", "--checkpoint", type=str, help="path to load checkpoint")
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA available: {}".format(torch.cuda.is_available()))
    print("CUDA device count: {}".format(torch.cuda.device_count()))
    print("CUDA device name: {}".format(torch.cuda.get_device_name(0)))
    print("CUDA current device: {}".format(torch.cuda.current_device()))
    args = parse_args()
    if args.eval:
        # python main.py --data path/to/data/ --category bottle --eval --check_points path/to/pt_file
        start = time.time()
        evaluate(args)
        end = time.time()
        print("Total time: {}".format(end - start))
    else:
        # python main.py --data path/to/data/ --category bottle
        start = time.time()
        train(args)
        end = time.time()
        print("Total time: {}".format(end - start))
