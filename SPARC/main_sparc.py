                                                                         

import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
import copy
import sys
import math
from pathlib import Path
from torch.nn import functional as F
import warnings

from timm.data import Mixup
from timm.models import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler

                                
import SDT2Net.EAB_FTF_vit_fixed

           
from sparc import SPARC

from datasets import build_dataset
from engine import evaluate
import utils
from utils import MultiEpochsDataLoader

warnings.filterwarnings('ignore')


def apply_ctcr_to_soft_targets(targets, current_task_class_offset):
                                     
    if current_task_class_offset == 0:
        return targets
    targets[:, :current_task_class_offset] = 0.0
    return targets / targets.sum(dim=1, keepdim=True)


def compute_cil_metrics(accuracy_matrix, stage_id):
                                                                       

                                                                     
                                                                         
                                                          
       
    current_task_accuracies = accuracy_matrix[stage_id, :stage_id + 1]
    if np.isnan(current_task_accuracies).any():
        raise ValueError(
            f"Missing task accuracy before computing metrics at stage {stage_id}."
        )

    macc = float(np.mean(current_task_accuracies))
    if stage_id == 0:
        return macc, None

    learned_task_diagonal = np.diag(accuracy_matrix)[:stage_id]
    final_old_task_accuracies = accuracy_matrix[stage_id, :stage_id]
    if (
        np.isnan(learned_task_diagonal).any()
        or np.isnan(final_old_task_accuracies).any()
    ):
        raise ValueError(
            f"Incomplete accuracy matrix before computing BWT at stage {stage_id}."
        )

    bwt = float(np.mean(
        learned_task_diagonal - final_old_task_accuracies
    ))
    return macc, bwt


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    loss_scaler, max_norm, mixup_fn, set_training_mode=True,
                    previous_stage_model=None, lambda_fd=0.0,
                    logger=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

                                       
            sparc = model.module if hasattr(model, 'module') else model
            targets = apply_ctcr_to_soft_targets(
                targets, sparc.current_task_class_offset
            )

        with torch.cuda.amp.autocast(enabled=False):
                       
            outputs = model(samples)
            loss_cls = criterion(outputs, targets)

            loss_csfd = torch.tensor(0.0).to(device)

            if previous_stage_model is not None and lambda_fd > 0:
                with torch.no_grad():
                    previous_stage_features = (
                        previous_stage_model.forward_prompt_expert_features(samples)
                    )

                current_stage_model = (
                    model.module if hasattr(model, 'module') else model
                )
                current_stage_features = (
                    current_stage_model.forward_prompt_expert_features(samples)
                )
                loss_csfd = (
                    F.mse_loss(
                        current_stage_features, previous_stage_features
                    )
                    * lambda_fd
                )

            loss = loss_cls + loss_csfd

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(loss_csfd=loss_csfd.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_args_parser():
    parser = argparse.ArgumentParser('SPARC Exemplar-Free CIL', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--num-tasks', default=None, type=int)
    parser.add_argument('--classes-per-task', default=7, type=int)

                                                
    parser.add_argument('--num-prompts', default=10, type=int)
    parser.add_argument('--expert-rank', default=16, type=int)

                                             
    parser.add_argument('--lambda-fd', default=1.0, type=float)

           
    parser.add_argument('--model', default='vit_deit_SDT2Net_small_patch16_224', type=str)
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    
                  
    parser.add_argument('--color-jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--mixup', type=float, default=0.8)
    parser.add_argument('--cutmix', type=float, default=1.0)
    parser.add_argument('--train-interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)

          
    parser.add_argument('--finetune', default='', type=str)
    parser.add_argument('--data-path', default='', type=str)
    parser.add_argument('--data-set', default='UCM', type=str)
    parser.add_argument('--output_dir', default='./log/ucm_sparc_exemplar_free/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true', default=True)
    return parser


def get_num_classes_and_tasks(args):
    dataset_classes = {
        'CIFAR': 100, 'IMNET': 1000, 'UCM': 21, 'AID': 30,
        'RSI-CB256': 35, 'NWPU': 45
    }
    total_classes = dataset_classes.get(args.data_set, 100)
    class_order = list(range(total_classes))
    if args.seed > 0:
        random.Random(args.seed).shuffle(class_order)

    tasks = []
    if args.classes_per_task:
        for i in range(0, total_classes, args.classes_per_task):
            tasks.append(class_order[i: i + args.classes_per_task])
    else:
        tasks = [class_order]
    return total_classes, tasks


def expand_task_classifier(model, new_classes_count, device):
    if hasattr(model, 'module'):
        model_ptr = model.module
    else:
        model_ptr = model

    if hasattr(model_ptr, 'expand_classifier'):
        model_ptr.expand_classifier(new_classes_count, device)
    elif hasattr(model_ptr, 'backbone') and hasattr(model_ptr.backbone, 'increment_classes'):
        model_ptr.backbone.increment_classes(new_classes_count, device)
    else:
        raise AttributeError("SPARC must implement 'expand_classifier'.")
    return model


def main(args):
    utils.init_distributed_mode(args)
    logger = utils.create_logger(Path(args.output_dir), dist_rank=utils.get_rank())
    logger.info(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    _, task_lists = get_num_classes_and_tasks(args)
    if args.num_tasks is None: args.num_tasks = len(task_lists)
    logger.info(f"Task Splits: {task_lists}")

    class_to_idx = {}
    current_model_dim = 0
    previous_stage_model = None
                                                                       
    accuracy_matrix = np.full(
        (args.num_tasks, args.num_tasks), np.nan, dtype=np.float64
    )

    for task_id in range(args.num_tasks):
        current_classes = task_lists[task_id]
        
        for cls in current_classes:
            if cls not in class_to_idx:
                class_to_idx[cls] = current_model_dim
                current_model_dim += 1
        
        logger.info(f"Task {task_id} | Classes: {current_classes} | Model Dim: {current_model_dim}")

                    
        dataset_train, _ = build_dataset(True, args, current_classes, class_to_idx)
        dataset_val, _ = build_dataset(False, args, current_classes, class_to_idx)
        
        loader_train = MultiEpochsDataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
        loader_val = MultiEpochsDataLoader(dataset_val, batch_size=int(1.5 * args.batch_size), shuffle=False,
                                           num_workers=args.num_workers, pin_memory=args.pin_mem)

                        
        if task_id == 0:
            logger.info("Creating base model...")
            base_model = create_model(
                args.model, pretrained=not bool(args.finetune),
                num_classes=len(current_classes),
            )
            if args.finetune:
                logger.info(f"Loading pretrained: {args.finetune}")
                checkpoint = torch.load(args.finetune, map_location='cpu')
                base_model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)

            logger.info("Creating SPARC with SPEA, SPIC, and DPHC...")
            model = SPARC(
                base_model,
                num_prompts=args.num_prompts,
                rank=args.expert_rank,
            )
            
            model.to(device)
            
            logger.info(f"[SPEA] Adding prompt-expert branch for Task {task_id}")
            model.add_task_branch(task_id, device)
        else:
                                     
            logger.info("[CSFD] Freezing previous-stage model...")
            model.cpu()
            previous_stage_model = copy.deepcopy(model)
            previous_stage_model.eval()
            for p in previous_stage_model.parameters():
                p.requires_grad = False
            previous_stage_model.to(device)
            logger.info(
                f"[CSFD] Previous stage has "
                f"{previous_stage_model.num_classes} classes."
            )
            
                       
            model.to(device)
            model = expand_task_classifier(
                model, len(current_classes), device
            )
            
            logger.info(f"[SPEA] Adding prompt-expert branch for Task {task_id}")
            model.add_task_branch(task_id, device)

        logger.info(f"[Main] Set model to Task {task_id} for training")
        model_without_ddp = model.module if hasattr(model, 'module') else model

                      
        parameters_to_train = [p for p in model_without_ddp.parameters() if p.requires_grad]
        logger.info(f"Training {len(parameters_to_train)} parameter groups")
        
        optimizer = torch.optim.AdamW(parameters_to_train, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, warmup_t=args.warmup_epochs, warmup_lr_init=1e-6)
        criterion = SoftTargetCrossEntropy() if args.mixup > 0 else torch.nn.CrossEntropyLoss()
        
        mixup_fn = None
        if args.mixup > 0:
            mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                             num_classes=current_model_dim, label_smoothing=args.smoothing)

                     
        logger.info(f"Start training Task {task_id}")
        for epoch in range(args.epochs):
            train_one_epoch(
                model, criterion, loader_train, optimizer, device, epoch,
                utils.NativeScalerWithGradNormCount(), 1.0, mixup_fn,
                previous_stage_model=previous_stage_model,
                lambda_fd=args.lambda_fd,
                logger=logger
            )
            lr_scheduler.step(epoch)
            
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                val_stats = evaluate(loader_val, model, device, logger=None)
                logger.info(f" Ep {epoch} Val Acc: {val_stats['acc1']:.2f}%")

                                                                    
                                                    
                                                                    
        logger.info(
            f"[DPHC] Constructing dual-space prototypes for Task {task_id}..."
        )
        
                                            
        _, base_transform = build_dataset(False, args, current_classes, class_to_idx)
        
                     
        ds_pure_for_proto, _ = build_dataset(True, args, current_classes, class_to_idx)
        
                                                                
        if hasattr(ds_pure_for_proto, 'transform'):
            ds_pure_for_proto.transform = base_transform
        elif hasattr(ds_pure_for_proto, 'dataset') and hasattr(ds_pure_for_proto.dataset, 'transform'):
            ds_pure_for_proto.dataset.transform = base_transform
            
        ld_proto = torch.utils.data.DataLoader(
            ds_pure_for_proto, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=args.pin_mem
        )
        model_without_ddp.construct_dual_space_prototypes(ld_proto, device)
                                                                    

                       
                                                                    
                   
                                                                    
        logger.info(f"[Main] Starting Evaluation up to Task {task_id}...")
        
                                     
        logger.info("--- Individual Task Accuracies ---")
        for t in range(task_id + 1):
            ds_t, _ = build_dataset(False, args, task_lists[t], class_to_idx)
            ld_t = torch.utils.data.DataLoader(
                ds_t, batch_size=args.batch_size, shuffle=False, 
                num_workers=args.num_workers, pin_memory=args.pin_mem
            )
            val_stats_t = evaluate(ld_t, model, device, logger=None)
            task_accuracy = float(val_stats_t['acc1'])
            accuracy_matrix[task_id, t] = task_accuracy
            logger.info(f" 👉 Task {t} Acc: {task_accuracy:.2f}%")

        stage_macc, stage_bwt = compute_cil_metrics(
            accuracy_matrix, task_id
        )
        logger.info(
            f" 🎯 mACC after Task {task_id}: {stage_macc:.2f}%"
        )
        if stage_bwt is None:
            logger.info(" 🎯 BWT after Task 0: N/A (no previous task)")
        else:
            logger.info(
                f" 🎯 BWT after Task {task_id}: {stage_bwt:.2f}% "
                "(positive means forgetting)"
            )

        if args.output_dir and utils.is_main_process():
            np.savetxt(
                Path(args.output_dir) / 'accuracy_matrix.csv',
                accuracy_matrix,
                delimiter=',',
                fmt='%.4f',
            )
            
                                       
        logger.info("--- Joint CIL Accuracy ---")
        all_seen_classes = []
        for t in range(task_id + 1):
            all_seen_classes.extend(task_lists[t])
            
        ds_joint, _ = build_dataset(False, args, all_seen_classes, class_to_idx)
        ld_joint = torch.utils.data.DataLoader(
            ds_joint, batch_size=args.batch_size * 2, shuffle=False, 
            num_workers=args.num_workers, pin_memory=args.pin_mem
        )
        
        joint_stats = evaluate(ld_joint, model, device, logger=None)
        joint_acc = joint_stats['acc1']
        
        logger.info(
            f" 🔥 Joint Accuracy (Task 0 to {task_id}): "
            f"{joint_acc:.2f}%"
        )

        if task_id == args.num_tasks - 1:
            logger.info("=== Final Class-Incremental Metrics ===")
            logger.info(f"Final mACC: {stage_macc:.2f}%")
            if stage_bwt is not None:
                logger.info(
                    f"Final BWT: {stage_bwt:.2f}% "
                    "(positive means forgetting)"
                )
                                                                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir: Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
