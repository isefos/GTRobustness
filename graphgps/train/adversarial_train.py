import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.train.custom_train import _get_epoch_log_results, eval_epoch, homophily_regularization
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name


def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        #batch.split = 'train' -> commented to make homophily_regularization possible
        batch.to(torch.device(cfg.accelerator))
        original_edge_index = batch.edge_index.clone()
        pred, true = model(batch)

        if cfg.gnn.head == "node" and batch.get("train_mask") is not None:
            pred_full, true_full = pred, true
            mask = batch.train_mask
            pred, true = pred_full[mask], true_full[mask] if true_full is not None else None
        else:
            pred_full, true_full = None, None

        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        
        # add homophily regularization if specified
        if pred_full is not None and cfg.train.homophily_regularization > 0:
            reg = homophily_regularization(
                pred_full, true_full, original_edge_index, batch.train_mask, batch.x.size(0),
            )
            loss += cfg.train.homophily_regularization * reg
        
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
        )
        time_start = time.time()


@register_train('adversarial')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    max_epoch = cfg.optim.max_epoch
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if start_epoch == max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    early_stopping = cfg.optim.early_stopping
    patience = cfg.optim.early_stopping_patience
    best_val_loss = None
    patience_e = cfg.optim.early_stopping_delta_e
    patience_warmup = start_epoch + cfg.optim.early_stopping_warmup

    # for polynormer with local pre-training:
    if cfg.optim.num_local_epochs > 0:
        assert max_epoch > cfg.optim.num_local_epochs, (
            f"total number of epochs ({max_epoch}) must be larger than "
            f"the number of local pre-train epochs ({cfg.optim.num_local_epochs})"
        )
        logging.info('local-only pre-training for %s epochs', cfg.optim.num_local_epochs)
        logging.info('global training for %s epochs', max_epoch - cfg.optim.num_local_epochs)
        patience_warmup += cfg.optim.num_local_epochs
        model.model._global = False

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, max_epoch):

        if early_stopping and patience <= 0:
            logging.info('Early stopping because validation loss is not decreasing further')
            break

        # for polynormer, with from local pre-training to global
        if cur_epoch == cfg.optim.num_local_epochs:
            logging.info('Local-only pre-training done, starting global training')
            model.model._global = True

        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model, split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            val_losses = np.array([vp['loss'] for vp in val_perf])
            val_loss_cur_epoch = float(val_losses[-1])

            if cur_epoch > patience_warmup:
                if best_val_loss is None:
                    best_val_loss = val_loss_cur_epoch
                elif val_loss_cur_epoch < best_val_loss:
                    best_val_loss = val_loss_cur_epoch
                elif (1 + patience_e) * best_val_loss <= val_loss_cur_epoch:
                    patience -= 1

            best_epoch = int(val_losses.argmin())
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                val_metric = np.array([vp[m] for vp in val_perf])
                if cfg.metric_agg == "argmax":
                    metric_agg = "max"
                elif cfg.metric_agg == "argmin":
                    metric_agg = "min"
                else:
                    raise ValueError("cfg.metric_agg should be either 'argmax' or 'argmin'")
                best_val_metric = getattr(val_metric, metric_agg)()
                best_val_metric_epochs = np.arange(len(val_metric), dtype=int)[val_metric == best_val_metric]
                # use loss to decide when epochs have same best metric
                best_epoch = int(best_val_metric_epochs[val_losses[best_val_metric_epochs].argmin()])
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None
    logging.info('Task done, results saved in %s', cfg.run_dir)
    # return the logged results per epoch in a differently organized way:
    results = _get_epoch_log_results(perf, best_epoch)
    return results
