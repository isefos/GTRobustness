import logging
import time

import numpy as np
import torch
from torch import Tensor
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.train.custom_train import (
    _get_epoch_log_results,
    eval_epoch,
    get_best_epoch,
    init_wandb,
    log_wandb_val_epoch,
    checkpoint_and_log,
)
from graphgps.utils import flatten_dict

from graphgps.attack.preprocessing import add_node_prob, remove_isolated_components
from torch_geometric.data import Batch, Data
from graphgps.attack.attack import get_attack_graph
from graphgps.attack.dataset_attack import get_total_dataset_graphs
from graphgps.transform.lap_eig import get_repeated_eigenvalue_slices
from graphgps.attack.prbcd import PRBCDAttack
from graphgps.attack.prbcd_nia import PRBCDAttackNI


@register_train('adversarial')
def adversarial_train(loggers, loaders, model, optimizer, scheduler):
    """Adversarial training pipeline."""
    # currently no node injection supported for transductive
    if cfg.dataset.task == "node" and (cfg.attack.node_injection.enable or cfg.attack.remove_isolated_components):
        raise NotImplementedError(
            "Need to handle the node mask (also to calculate attack success rate) "
            "with node injection or pruning away isolated components."
        )
    start_epoch = 0
    # should be set as the number of "normal" epochs divided by the number of replays
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
    patience_m = 1 + cfg.optim.early_stopping_delta_e
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

    # for the attack part
    if cfg.attack.node_injection.enable:
        AttackBaseClass = PRBCDAttackNI
    else:
        AttackBaseClass = PRBCDAttack
    TrainAttackClass = get_train_attack_class(AttackBaseClass)
    
    if cfg.wandb.use:
        run = init_wandb()

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, max_epoch):

        if early_stopping and patience <= 0:
            logging.info('Early stopping because validation loss is not decreasing further')
            break

        # for polynormer, with from local pre-training to global
        if cur_epoch and cur_epoch == cfg.optim.num_local_epochs:
            logging.info('Local-only pre-training done, starting global training')
            model.model._global = True

        # train epoch
        start_time = time.perf_counter()
        adversarial_train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, TrainAttackClass)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            # TODO: add attack val eval
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model, split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            # repeat log of previous eval
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        # lr scheduler
        if cfg.optim.scheduler == 'reduce_on_plateau':
            # based on val loss
            scheduler.step(perf[1][-1]['loss'])
        else:
            scheduler.step()

        # time
        full_epoch_times.append(time.perf_counter() - start_time)

        # checkpoint with regular frequency (if enabled)
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # log current best stats on eval epoch
        if is_eval_epoch(cur_epoch):
            val_losses = np.array([vp['loss'] for vp in perf[1]])
            val_loss_cur_epoch = float(val_losses[-1])

            # patience is computed on validation loss
            if cur_epoch > patience_warmup:
                if best_val_loss is None:
                    best_val_loss = val_loss_cur_epoch
                elif val_loss_cur_epoch < best_val_loss:
                    best_val_loss = val_loss_cur_epoch
                elif patience_m * best_val_loss <= val_loss_cur_epoch:
                    patience -= 1

            best_epoch = int(val_losses.argmin())
            best_m = ["", "", ""]  # loss is reported anyway

            if cfg.metric_best != 'auto':
                # select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                
                # also updated best_m with the metric values
                best_epoch = get_best_epoch(perf, val_losses, m, best_m)

                if cfg.wandb.use:
                    log_wandb_val_epoch(run, perf, m, cur_epoch, best_epoch, full_epoch_times)

            # checkpoint and log
            checkpoint_and_log(model, optimizer, scheduler, cur_epoch, best_epoch, perf, full_epoch_times, best_m)
    
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    if cfg.wandb.use:
        run.finish()

    logging.info('Task done, results saved in %s', cfg.run_dir)
    # return the logged results per epoch (organized differently)
    results = _get_epoch_log_results(perf, best_epoch)
    return results


def adversarial_train_epoch(logger, loader, model, optimizer, scheduler, TrainAttackClass):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for b_i, batch in enumerate(loader):
        # Each batch can be seen as the dataset to be attacked 
        # (batch shuffle=True, therefore can't create constant global index)
        # Do we also want to be able to insert nodes from other batches? 
        #  - With shuffle not that necessary, given that batch size is large enough

        # for node injection attacks, precompute the dataset / graph augmentations
        total_attack_dataset_graph, attack_dataset_slices = None, None
        if cfg.attack.node_injection.enable:
            total_attack_dataset_graph, attack_dataset_slices, _ = get_total_dataset_graphs(
                inject_nodes_from_attack_dataset=True,
                dataset_to_attack=batch,
                additional_injection_datasets=None,
                include_root_nodes=cfg.attack.node_injection.include_root_nodes,
            )
        batch_size = batch.num_graphs

        # create an attacker for each graph in the batch (currently all loaded onto GPU, maybe inefficient)
        attackers = []
        for i, data in enumerate(batch.to_data_list()):
            # get the train nodes' indices (currently incompatible with node injection, checked at start of training)
            node_mask = data.get(f'train_mask')
            if node_mask is not None:
                node_mask = node_mask.to(device=torch.device(cfg.accelerator))
                assert not cfg.attack.prediction_level == "graph"
            # augment graph when node injection
            attack_graph_data = get_attack_graph(
                graph_data=data,
                total_attack_dataset_graph=total_attack_dataset_graph,
                attack_dataset_slice=None if attack_dataset_slices is None else attack_dataset_slices[i],
                # only inject from training dataset to prevent val/test data leakage
                total_additional_datasets_graph=None,
            )
            attack_graph_data.to(device=torch.device(cfg.accelerator))
            # create new attack object for this graph
            attackers.append(
                TrainAttackClass(
                    model,
                    x=attack_graph_data.x,
                    edge_index=attack_graph_data.edge_index,
                    labels=attack_graph_data.y,
                    idx_attack=node_mask,
                )
            )

        # replay (train multiple iterations) on the same batch k times to get a k-step attack optimization 
        for replay in range(cfg.train.adv.num_replays):
            data_list = []

            # loop over batch samples and do 1 attack step and get 1 new adversarial sample
            for d_i, train_attack in enumerate(attackers):
                # optimization step for attack
                attack_step_stats = train_attack.train_attack_step(replay)
                # TODO: log these stats?
                data = train_attack.get_discrete_sample()
                # for train
                if cfg.train.adv.batched_train:
                    # save adv sample for batched update
                    data_list.append(data)
                else:
                    # already do single update and accumulate gradient
                    time_start = train_forward_backward(model, data, batch_size, time_start, logger, scheduler)

            if cfg.train.adv.batched_train:
                time_start = train_forward_backward(model, data_list, batch_size, time_start, logger, scheduler)
            # model parameter update step outside of the batch samples loop (with accumulated gradients)
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad()


def train_forward_backward(model, data: list[Data] | Data, batch_size: int, time_start, logger, scheduler):
    if type(data) is list:
        data = Batch.from_data_list(data)
        batch_size = 1  # no need for batch normalization, already doing batched update
        retain_graph = False
    else:
        data = Batch.from_data_list([data])
        retain_graph = True
    data.split = 'train'
    # forward train
    pred, true = model(data)
    # training loss and backward
    if cfg.dataset.name == 'ogbg-code2':
        loss, pred_score = subtoken_cross_entropy(pred, true)
        _true = true
        _pred = pred_score
    else:
        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
    # when accumulating the model param gradients: normalize by batch size
    if batch_size > 1:
        loss /= batch_size
    loss.backward(retain_graph=retain_graph)
    # TODO: update other attack related stats?
    # log unnormalized loss, because logger also normalizes, we avoid double norm to be comparable to val loss
    logger.update_stats(
        true=_true,
        pred=_pred,
        loss=loss.detach().cpu().item() * batch_size,
        lr=scheduler.get_last_lr()[0],
        time_used=time.time() - time_start,
        params=cfg.params,
        dataset_name=cfg.dataset.name,
    )
    return time.time()


def get_train_attack_class(AttackBaseClass):
    """
    returns attack class with some new / overwritten methods for "free" adversarial training setting
    """
    assert AttackBaseClass in [PRBCDAttack, PRBCDAttackNI]

    class TrainAttackClass(AttackBaseClass):
        def __init__(
            self,
            model,
            x: Tensor,
            edge_index: Tensor,
            labels: Tensor,
            idx_attack: None | Tensor = None,
        ):
            super().__init__(model)
            # Set early stopping to False
            self.coeffs['with_early_stopping'] = False

            # compute the budget, at least 1
            budget_edges = edge_index.size(1)
            if cfg.attack.is_undirected:
                budget_edges //= 2
            # at least one
            self.budget = max(1, int(cfg.train.adv.e_budget * budget_edges))
            
            self.block_size = cfg.train.adv.block_size
            self.epochs = cfg.train.adv.num_replays
            self.lr = cfg.train.adv.lr
            # always resample
            self.epochs_resampling = self.epochs
            self.resample_period = 1

            # set eps higher, such that early iterations can already produce num perturbations close to budget
            # init such that 10% of the budget is used
            self.coeffs['eps'] = 0.1 * self.budget / self.block_size

            # prepare for the attack steps
            self._attack_self_setup(x, edge_index)
            # if x is on cpu will set self.device to cpu, could either:
            #  1) set self.device = cfg.accelerator  # here, then the perturbation tensors will all be on gpu
            #  2) move everything onto gpu only when needed in the forward pass (may take more time, but avoids 
            #     memory issues for large batch sizes with large block sizes)
            #  3) move x (and others) to gpu before creating, could potentially use a lot of GPU memory 
            #     (each graph in the batch could be augmented to the size of the batch, so b^2 instead of b memory)
            # for now chose 3)
            #self.device = cfg.accelerator
            self._setup_sampling(x=x)
            self.x = x
            self.labels = labels
            self.idx_attack = idx_attack
            _ = self._prepare(self.budget)
            # only set model to eval when actually inside attack step method
            model.train()

        def train_attack_step(self, replay: int):
            self.model.eval()
            # don't want to accumulate training gradients from this attack forward pass
            for param in self.model.parameters():
                param.requires_grad = False
            # do the attack optimization
            loss, gradient = self._forward_and_gradient(self.x, self.labels, self.idx_attack, retain_graph=True)
            scalars = self._update(replay, gradient, self.x, self.labels, self.budget, loss, self.idx_attack, None)
            # reset the gradient for next step
            self.block_edge_weight.grad = None
            self.block_edge_weight.requires_grad = False
            # accumulate training gradients from next train forward pass
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
            return scalars
        
        def _forward(self, x: Tensor, edge_index: None | Tensor, edge_weight: Tensor, discrete: bool, **kwargs):
            assert (discrete and edge_weight is None) or (not discrete and edge_weight is not None)
            data = self._get_forward_data(x, edge_index, edge_weight, discrete)
            # remove isolated components (if specified in cfg), important for efficient node injection
            data, root_node = remove_isolated_components(data)
            # add the "clean" laplacian info, from which will be perturbed
            if not discrete and cfg.posenc_WLapPE.enable and cfg.attack.SAN.enable_pert_grad:
                self._add_laplacian_info(data)
                # check for repeated eigenvalues:
                data.E_rep_slices_min, data.E_rep_slices_max = get_repeated_eigenvalue_slices(
                    data.E_clean, cfg.attack.SAN.eps_repeated_eigenvalue,
                )
            data.attack_mode = True
            # add node probability, if needed
            add_node_prob(data, root_node)
            data = Batch.from_data_list([data])
            model_prediction, _ = self.model(data, **kwargs)  # don't need the y ground truth
            return model_prediction

        def get_discrete_sample(self) -> Data:
            """only a single sample without budget check for a single training forward pass"""
            block_edge_weight = self.block_edge_weight
            block_edge_weight[block_edge_weight <= self.coeffs['eps']] = 0
            sampled_edges = torch.zeros_like(block_edge_weight)
            sampled_edges_mask = torch.bernoulli(block_edge_weight).to(bool)
            sampled_edges[sampled_edges_mask] = block_edge_weight[sampled_edges_mask]
            # no budget check, not too important during training
            edge_index = self._get_discrete_sampled_graph(sampled_edges)[0]
            data = self._get_forward_data(self.x, edge_index, edge_weight=None, discrete=True)
            # add back the train_mask
            if self.idx_attack:
                data.train_mask = self.idx_attack
            # remove isolated components (if specified in cfg), important for efficient node injection
            data, root_node = remove_isolated_components(data)
            data.attack_mode = False
            # add node probability, if needed
            add_node_prob(data, root_node)
            # add y
            data.y = self.labels
            return data


    return TrainAttackClass
