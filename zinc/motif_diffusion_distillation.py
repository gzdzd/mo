import csv
import math
import os
import random
import time
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import Net, NetL
from data_preparation import MoleculeDataset
from utils.config import get_args, process_config


torch.set_num_threads(1)


def build_model(config, num_basis, atom_dim, bond_dim):
    if config.hyperparams.get('model', 'Net') == 'NetL':
        return NetL(
            config.architecture,
            num_tasks=1,
            num_basis=num_basis,
            atom_dim=atom_dim,
            bond_dim=bond_dim,
        )
    return Net(
        config.architecture,
        num_tasks=1,
        num_basis=num_basis,
        shared_filter=config.architecture.get('shared_filter', '') == 'shd',
        linear_filter=config.architecture.get('linear_filter', '') == 'lin',
        atom_dim=atom_dim,
        bond_dim=bond_dim,
    )


def split_labeled_unlabeled(dataset, labeled_ratio, seed):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    labeled_count = max(1, int(len(indices) * labeled_ratio))
    labeled_indices = indices[:labeled_count]
    unlabeled_indices = indices[labeled_count:]
    return Subset(dataset, labeled_indices), Subset(dataset, unlabeled_indices)


def ema_update(teacher, student, decay):
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)


def infinite_loader(loader):
    for batch in cycle(loader):
        yield batch


def train_distillation_epoch(
    student,
    teacher,
    device,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    unsup_weight,
    ema_decay,
    max_grad_norm=None,
):
    student.train()
    teacher.eval()

    loss_all = 0.0
    distill_all = 0.0
    sup_all = 0.0

    unlabeled_iter = infinite_loader(unlabeled_loader) if unlabeled_loader is not None else None

    for step, (bg, labels) in enumerate(tqdm(labeled_loader, desc="Train iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop('feat')
        edge_attr = bg.edata.pop('feat')
        bases = bg.edata.pop('bases')
        labels = labels.to(device)

        pred = student(bg, x, edge_attr, bases)
        sup_loss = F.l1_loss(pred, labels)
        distill_loss = torch.tensor(0.0, device=device)

        if unlabeled_iter is not None and unsup_weight > 0:
            unlab_bg, _ = next(unlabeled_iter)
            unlab_bg = unlab_bg.to(device)
            unlab_x = unlab_bg.ndata.pop('feat')
            unlab_edge_attr = unlab_bg.edata.pop('feat')
            unlab_bases = unlab_bg.edata.pop('bases')

            with torch.no_grad():
                teacher_pred = teacher(unlab_bg, unlab_x, unlab_edge_attr, unlab_bases)

            student_pred = student(unlab_bg, unlab_x, unlab_edge_attr, unlab_bases)
            distill_loss = F.l1_loss(student_pred, teacher_pred)

        loss = sup_loss + unsup_weight * distill_loss

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
        optimizer.step()
        ema_update(teacher, student, ema_decay)

        loss_all += loss.detach().item()
        distill_all += distill_loss.detach().item()
        sup_all += sup_loss.detach().item()

    steps = len(labeled_loader)
    return loss_all / steps, sup_all / steps, distill_all / steps


def eval_model(model, device, loader):
    model.eval()
    total_mae = 0.0
    with torch.no_grad():
        for step, (bg, labels) in enumerate(tqdm(loader, desc="Eval iteration")):
            bg = bg.to(device)
            x = bg.ndata.pop('feat')
            edge_attr = bg.edata.pop('feat')
            bases = bg.edata.pop('bases')
            labels = labels.to(device)
            pred = model(bg, x, edge_attr, bases)
            total_mae += F.l1_loss(pred, labels).detach().item()
    return total_mae / (step + 1)


def run_with_given_seed(config, run_tag):
    if config.hyperparams.get('seed') is not None:
        random.seed(config.hyperparams.seed)
        torch.manual_seed(config.hyperparams.seed)
        np.random.seed(config.hyperparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.hyperparams.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MoleculeDataset(name=config.dataset_name, config=config.preprocess)

    distill_cfg = config.get('distillation', {})
    labeled_ratio = distill_cfg.get('labeled_ratio', 0.1)
    ema_decay = distill_cfg.get('ema_decay', 0.99)
    unsup_weight = distill_cfg.get('unsup_weight', 1.0)
    warmup_epochs = distill_cfg.get('teacher_warmup', 5)
    max_grad_norm = distill_cfg.get('max_grad_norm')

    labeled_set, unlabeled_set = split_labeled_unlabeled(
        dataset.train,
        labeled_ratio=labeled_ratio,
        seed=config.hyperparams.seed,
    )

    train_loader = DataLoader(
        labeled_set,
        batch_size=config.hyperparams.batch_size,
        shuffle=True,
        num_workers=config.hyperparams.num_workers,
        collate_fn=dataset.collate,
    )
    unlabeled_loader = DataLoader(
        unlabeled_set,
        batch_size=config.hyperparams.batch_size,
        shuffle=True,
        num_workers=config.hyperparams.num_workers,
        collate_fn=dataset.collate,
    ) if len(unlabeled_set) > 0 else None

    valid_loader = DataLoader(
        dataset.val,
        batch_size=config.hyperparams.batch_size,
        shuffle=False,
        num_workers=config.hyperparams.num_workers,
        collate_fn=dataset.collate,
    )
    test_loader = DataLoader(
        dataset.test,
        batch_size=config.hyperparams.batch_size,
        shuffle=False,
        num_workers=config.hyperparams.num_workers,
        collate_fn=dataset.collate,
    )

    atom_dim = 28
    bond_dim = 4
    num_basis = dataset.train.graph_lists[0].edata['bases'].shape[1]
    student = build_model(config, num_basis, atom_dim, bond_dim).to(device)
    teacher = build_model(config, num_basis, atom_dim, bond_dim).to(device)
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.hyperparams.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.hyperparams.weight_decay,
    )
    warmup_epochs = config.hyperparams.warmup_epochs
    lr_plan = lambda cur_epoch: (cur_epoch + 1) / warmup_epochs if cur_epoch < warmup_epochs else \
        (0.5 * (1.0 + math.cos(math.pi * (cur_epoch - warmup_epochs) / (config.hyperparams.epochs - warmup_epochs))))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_plan)

    writer = SummaryWriter(config.directory + 'board/')

    epoch_idx = []
    train_curve = []
    valid_curve = []
    test_curve = []
    loss_curve = []
    distill_curve = []

    for epoch in range(1, config.hyperparams.epochs + 1):
        epoch_unsup_weight = unsup_weight if epoch > warmup_epochs else 0.0
        train_loss, sup_loss, distill_loss = train_distillation_epoch(
            student,
            teacher,
            device,
            train_loader,
            unlabeled_loader,
            optimizer,
            epoch_unsup_weight,
            ema_decay,
            max_grad_norm=max_grad_norm,
        )
        scheduler.step()

        train_perf = eval_model(student, device, train_loader)
        valid_perf = eval_model(student, device, valid_loader)
        test_perf = eval_model(student, device, test_loader)

        print(
            'Epoch:',
            epoch,
            'Train:',
            train_perf,
            'Validation:',
            valid_perf,
            'Test:',
            test_perf,
            'Train loss:',
            train_loss,
        )

        epoch_idx.append(epoch)
        train_curve.append(train_perf)
        valid_curve.append(valid_perf)
        test_curve.append(test_perf)
        loss_curve.append(train_loss)
        distill_curve.append(distill_loss)

        writer.add_scalars('traP', {run_tag: train_perf}, epoch)
        writer.add_scalars('valP', {run_tag: valid_perf}, epoch)
        writer.add_scalars('tstP', {run_tag: test_perf}, epoch)
        writer.add_scalars('traL', {run_tag: train_loss}, epoch)
        writer.add_scalars('traDistill', {run_tag: distill_loss}, epoch)
        writer.add_scalars('traSup', {run_tag: sup_loss}, epoch)
        writer.add_scalars('lr', {run_tag: scheduler.optimizer.param_groups[0]['lr']}, epoch)

    writer.close()

    return epoch_idx, train_curve, valid_curve, test_curve, loss_curve, distill_curve


def main():
    args = get_args()
    config = process_config(args)
    cuda_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0), cuda_id)
    print(config)

    algo_setting = (
        f"{config.commit_id[0:7]}_{cuda_id}"
        f"{config.hyperparams.get('model', '')}"
        f"{config.architecture.get('shared_filter', '')}"
        f"{config.architecture.get('linear_filter', '')}"
        f"{config.preprocess.get('edgehop', '')}"
        f"E{config.preprocess.get('norm', '')}"
        f"{config.preprocess.get('eigen', '')}"
        f"{config.preprocess.get('norm2', '')}"
        f"P{config.preprocess.get('power', '')}"
        f"{config.preprocess.get('aug_coeff', '')}"
        f"n{config.preprocess.get('aug_n', '')}"
        f"e{config.preprocess.get('aug_e', '')}"
        f"{config.architecture.layers}_{config.architecture.hidden}_"
        f"{config.hyperparams.learning_rate}_{config.hyperparams.warmup_epochs}_"
        f"{config.hyperparams.weight_decay}"
        f"B{config.hyperparams.batch_size}"
        f"W{config.hyperparams.get('num_workers', 'na')}"
    )

    algo_setting = algo_setting.replace(' ', '').replace('[', ':').replace(']', ':')
    csv_dir = config.directory + 'stat/'
    os.makedirs(os.path.dirname(csv_dir + algo_setting + '/'), exist_ok=True)
    path_stat_total = csv_dir + algo_setting + '/' + str(config.time_stamp) + 'stat_total.csv'
    with open(path_stat_total, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['ts_fk_algo_hp', 'seed', 'test', 'valid', 'best_val_epoch', 'best_train', 'min_train_loss'])
        csv_file.flush()

    for seed in config.hyperparams.seeds:
        config.hyperparams.seed = seed
        config.time_stamp = int(time.time())
        run_tag = algo_setting + '/T' + str(config.time_stamp) + '_S' + str(config.hyperparams.seed)

        epoch_idx, train_curve, valid_curve, test_curve, loss_curve, distill_curve = run_with_given_seed(
            config,
            run_tag,
        )

        with open(csv_dir + run_tag + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['epoch', 'train', 'valid', 'test', 'train_loss', 'distill_loss'])
            csv_writer.writerows(
                np.transpose(
                    np.array([epoch_idx, train_curve, valid_curve, test_curve, loss_curve, distill_curve])
                )
            )
            csv_file.flush()

        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)
        with open(path_stat_total, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    run_tag,
                    config.hyperparams.seed,
                    test_curve[best_val_epoch],
                    valid_curve[best_val_epoch],
                    best_val_epoch,
                    best_train,
                    min(loss_curve),
                ]
            )
            csv_file.flush()

    with open(path_stat_total, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        column_test = []
        column_valid = []
        for row in csv_reader:
            column_test.append(row['test'])
            column_valid.append(row['valid'])

        column_test = np.array(column_test, dtype=float)
        column_valid = np.array(column_valid, dtype=float)
        test_stat = str(np.mean(column_test)) + '_' + str(np.std(column_test))
        valid_stat = str(np.mean(column_valid)) + '_' + str(np.std(column_valid))

    with open(path_stat_total, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['', '', test_stat, valid_stat, '', '', ''])
        csv_file.flush()


if __name__ == "__main__":
    main()
