from __future__ import annotations

import argparse
import json
import os
import pathlib

from dataloader import BatchLoader
from model import (
    NnBoard768Cuda,
    NnBoard768,
    NnHalfKA,
    NnHalfKACuda,
    NnHalfKP,
    NnHalfKPCuda,
    NnBoard768Dropout
)
from time import time

import torch
from trainlog import TrainLog

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_ITERS = 100_000_000

class AdamWithMomentum(torch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super(AdamWithMomentum, self).__init__(*args, **kwargs)
        self.defaults['momentum'] = self.defaults["betas"][0]

class WeightClipper:
    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(-1.98, 1.98)
            module.weight.data = w


def train(
    scheduler,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: BatchLoader,
    wdl: float,
    scale: float,
    epochs: int,
    save_epochs: int,
    train_id: str,
    lr_drop: int | None = None,
    train_log: TrainLog | None = None,
    lastEpochs = 0,
) -> None:
    clipper = WeightClipper()
    running_loss = torch.zeros((1,), device=DEVICE)
    start_time = time()
    iterations = 0

    loss_since_log = torch.zeros((1,), device=DEVICE)
    iter_since_log = 0

    fens = 0
    epoch = lastEpochs


    if scheduler is None:
        if epoch >= lr_drop:
            optimizer.param_groups[0]["lr"] *= 0.1
            print(f"Dropping learning rate")

    while epoch < epochs:
        new_epoch, batch = dataloader.read_batch(DEVICE)
        if new_epoch:
            epoch += 1
            if scheduler is None:
                if epoch >= lr_drop:
                    optimizer.param_groups[0]["lr"] *= 0.1
                    print(f"Dropping learning rate")

            print(
                f"epoch {epoch}",
                f"epoch train loss: {running_loss.item() / iterations}",
                f"epoch pos/s: {fens / (time() - start_time)}",
                sep=os.linesep,
            )

            running_loss = torch.zeros((1,), device=DEVICE)
            start_time = time()
            iterations = 0
            fens = 0

            if epoch % save_epochs == 0:
                torch.save(model.state_dict(), f"nn/{train_id}_{epoch}")
                torch.save(optimizer.state_dict(), f"nn/opt{train_id}_{epoch}")

                if scheduler is not None:
                    torch.save(scheduler.state_dict(), f"nn/sch{train_id}_{epoch}")

                param_map = {
                    name: param.detach().cpu().numpy().tolist()
                    for name, param in model.named_parameters()
                }
                with open(f"nn/{train_id}.json", "w") as json_file:
                    json.dump(param_map, json_file)


        optimizer.zero_grad()
        prediction = model(batch)
        expected = torch.sigmoid(batch.cp / scale) * (1 - wdl) + batch.wdl * wdl

        loss = torch.mean((prediction - expected) ** 2)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.apply(clipper)

        with torch.no_grad():
            running_loss += loss
            loss_since_log += loss
        iterations += 1
        iter_since_log += 1
        fens += batch.size

        if iter_since_log * batch.size > LOG_ITERS:
            loss = loss_since_log.item() / iter_since_log
            print(
                f"At {iterations * batch.size} positions",
                f"Running Loss: {loss}",
                sep=os.linesep,
            )
            if train_log is not None:
                train_log.update(loss)
                train_log.save()
            iter_since_log = 0
            loss_since_log = torch.zeros((1,), device=DEVICE)


def main():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--data-root", type=str, help="Root directory of the data files"
    )
    parser.add_argument("--train-id", type=str, help="ID to save train logs with")
    parser.add_argument("--lr", type=float, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, help="Epochs to train for")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size")
    parser.add_argument("--wdl", type=float, default=0.0, help="WDL weight to be used")
    parser.add_argument("--scale", type=float, help="WDL weight to be used")
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=100,
        help="How often the program will save the network",
    )
    parser.add_argument(
        "--lr-drop",
        type=int,
        default=None,
        help="The epoch learning rate will be dropped",
    )
    parser.add_argument("--last-epoch",  type=int, help="Last epoch to continue from")
    
    args = parser.parse_args()

    assert args.train_id is not None
    assert args.scale is not None
    assert args.last_epoch is not None

    train_log = TrainLog(args.train_id)

    model = NnBoard768(256).to(DEVICE)

    data_path = pathlib.Path(args.data_root)
    paths = list(map(str, data_path.glob("*.bin")))
    dataloader = BatchLoader(paths, model.input_feature_set(), args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=0.01, cycle_momentum=False)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, cycle_momentum=False, steps_per_epoch=2840)

    lastEpochs = 0

    if os.path.exists(f"nn/{args.train_id}_{args.last_epoch}"):
        print(f"Loading nn from checkpoint..(Epoch: {args.last_epoch})")
        checkpoint = torch.load(f"./nn/{args.train_id}_{args.last_epoch}", map_location=DEVICE)

        model.load_state_dict(checkpoint)
        print("Loaded nn!")
        lastEpochs = args.last_epoch
    
    if os.path.exists(f"nn/opt{args.train_id}_{args.last_epoch}"):
        print(f"Loading optimizer from checkpoint..(Epoch: {args.last_epoch})")
        checkpoint = torch.load(f"./nn/opt{args.train_id}_{args.last_epoch}", map_location=DEVICE)
        optimizer.load_state_dict(checkpoint)
        print("Loaded optimizer!")

    if (os.path.exists(f"nn/sch{args.train_id}_{args.last_epoch}")):
        print(f"Loading scheduler from checkpoint.. (Epoch: {args.last_epoch})")
        checkpoint = torch.load(f"./nn/sch{args.train_id}_{args.last_epoch}", map_location=DEVICE)
        scheduler.load_state_dict(checkpoint)
        print("Loaded scheduler!")

    train(
        scheduler,
        model,
        optimizer,
        dataloader,
        args.wdl,
        args.scale,
        args.epochs,
        args.save_epochs,
        args.train_id,
        lr_drop=args.lr_drop,
        train_log=train_log,
        lastEpochs=lastEpochs
    )   

    print("Training finished!")


if __name__ == "__main__":
    main()
