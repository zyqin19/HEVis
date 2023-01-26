from argparse import ArgumentParser
from datetime import timedelta
from glob import glob

from stemseg.data.common import tensor_struct_to, collate_fn
from stemseg.config import cfg as global_cfg
from stemseg.modeling.model_builder import build_model
from stemseg.utils import ModelPaths, RepoPaths
from stemseg.utils import distributed as dist_utils

from stemseg.training.interrupt_detector import InterruptDetector, InterruptException
from stemseg.training.model_output_manager import ModelOutputManager
from stemseg.training.training_logger import TrainingLogger
from stemseg.training.utils import create_training_dataset, create_optimizer, create_lr_scheduler, var_keys_to_str, \
    create_training_data_loader, register_log_level_type

import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn

import warnings
import yaml

try:
    import torch.cuda.amp as amp
    APEX_IMPORTED = True
except ImportError as _:
    APEX_IMPORTED = False


class Trainer(object):
    def __init__(self, cfg, model_save_dir, args, logger):
        self.num_gpus = dist_utils.get_world_size()
        self.local_rank = dist_utils.get_rank()
        self.local_device = dist_utils.get_device()
        self.is_main_process = dist_utils.is_main_process()

        self.console_logger = logger

        self.model_save_dir = model_save_dir
        self.log_dir = os.path.join(self.model_save_dir, 'logs')

        if self.is_main_process:
            os.makedirs(self.log_dir, exist_ok=True)

        self.model = build_model(cfg, restore_pretrained_backbone_wts=True, logger=self.console_logger).to(self.local_device)
        self.model_init = args.model_init
        self.glow_iterations = 0

        # create optimizer
        self.optimizer = create_optimizer(self.model, cfg.TRAINING, self.console_logger.info)

        # wrap model and optimizer around apex if mixed precision training is enabled
        if cfg.TRAINING.MIXED_PRECISION and APEX_IMPORTED:
            self.amp = True
        else:
            self.amp = False
            print("Could not use torch.cuda.amp. Training env will be unavailable.")

        if dist_utils.is_distributed():
            self.model = nn.parallel.DistributedDataParallel(
                module=self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=True
            )

        self.total_iterations = cfg.TRAINING.MAX_ITERATIONS
        self.total_glow_iterations = cfg.TRAINING.GLOW_EPOCH * cfg.TRAINING.BATCH_SIZE

        # create LR scheduler
        self.lr_scheduler = create_lr_scheduler(self.optimizer, cfg.TRAINING, self.console_logger.info)

        # create parameter logger
        self.logger = None
        if self.is_main_process:
            self.logger = TrainingLogger(self.log_dir)

        self.interrupt_detector = InterruptDetector()
        self.cfg = cfg

        self.elapsed_iterations = 0

        assert not (args.restore_session and args.initial_ckpt)

        if args.restore_session:
            self.console_logger.info("Restoring session from {}".format(args.restore_session))
            self.restore_session(torch.load(args.restore_session, map_location=self.local_device))
        elif args.initial_ckpt:
            self.console_logger.info("Loading model weights from checkpoint at: {}".format(args.initial_ckpt))
            self._model.load_state_dict(torch.load(args.initial_ckpt, map_location=self.local_device)['model'])

    @property
    def _model(self):
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module
        else:
            return self.model

    def backup_session(self):
        model_save_path = os.path.join(self.model_save_dir, '{:06d}.pth'.format(self.elapsed_iterations))

        save_dict = {'model': self._model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'lr_scheduler': self.lr_scheduler.state_dict(),
                     'logger': self.logger.state_dict(),
                     'iterations': self.elapsed_iterations,
                     'glow_iterations': self.glow_iterations}

        # if global_cfg.TRAINING.MIXED_PRECISION:
        #     save_dict['amp'] = amp.state_dict()

        torch.save(save_dict, model_save_path, _use_new_zipfile_serialization=False)
        self.console_logger.info("Checkpoint saved to: {}".format(model_save_path))
        return model_save_path

    def restore_session(self, restore_dict):
        assert 'model' in restore_dict, "Restore state dict contains no entry named 'model'"
        if self.model_init:
            model_dict = self.model.state_dict()
            state_dict = {k: v for k, v in restore_dict['model'].items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self._model.load_state_dict(model_dict)
        else:
            self._model.load_state_dict(restore_dict['model'])

        assert 'optimizer' in restore_dict, "Restore state dict contains no entry named 'optimizer'"
        if not self.model_init:
            self.optimizer.load_state_dict((restore_dict['optimizer']))

        if not self.model_init:
            assert 'lr_scheduler' in restore_dict, "Restore state dict contains no entry named 'lr_scheduler'"
            self.lr_scheduler.load_state_dict(restore_dict['lr_scheduler'])

        if not self.model_init:
            assert 'iterations' in restore_dict, "Restore state dict contains no entry named 'iterations'"
            self.elapsed_iterations = restore_dict['iterations']

        if not self.model_init:
            assert 'glow_iterations' in restore_dict, "Restore state dict contains no entry named 'glow_iterations'"
            self.glow_iterations = restore_dict['glow_iterations']

        # if 'amp' in restore_dict and global_cfg.TRAINING.MIXED_PRECISION:
        #     amp.load_state_dict(restore_dict['amp'])

        if self.is_main_process:
            assert 'logger' in restore_dict, "Restore state dict contains no entry named 'logger'"
            self.logger.load_state_dict(restore_dict['logger'])

    def start(self, opts):
        max_samples_per_gpu = self.cfg.TRAINING.MAX_SAMPLES_PER_GPU
        batch_size = self.cfg.TRAINING.BATCH_SIZE
        accumulate_gradients = self.cfg.TRAINING.ACCUMULATE_GRADIENTS

        dataset = create_training_dataset(self.total_iterations * batch_size, print_fn=self.console_logger.info)

        if accumulate_gradients:
            assert batch_size >= self.num_gpus, "Batch size ({}) must be >= number of GPUs ({})".format(
                batch_size, self.num_gpus)

            optimizer_step_interval = int(batch_size / (max_samples_per_gpu * self.num_gpus))
            assert batch_size % max_samples_per_gpu == 0, \
                "Batch size ({}) must be divisible by number of samples per GPU ({})".format(
                    batch_size, max_samples_per_gpu)

            if self.is_main_process:
                self.console_logger.info("Optimizer will be run every {} iterations".format(optimizer_step_interval))
        else:
            if batch_size > max_samples_per_gpu:
                raise ValueError("A batch size of {} cannot be processed. Max samples per GPU = {}".format(
                    batch_size, max_samples_per_gpu))

            max_samples_per_gpu = batch_size
            optimizer_step_interval = 1

        if self.is_main_process:
            n_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.console_logger.info(
                "Commencing/resuming training with the following settings:\n"
                "- Elapsed iterations: %d\n"
                "- Total iterations: %d\n"
                "- Total glow iterations: %d\n"
                "- Batch size: %d\n"
                "- Optimizer step interval: %d\n"
                "- Model save directory: %s\n"
                "- Save interval: %d\n"
                "- Trainable parameters: %d" % (
                    self.elapsed_iterations, self.total_iterations, self.total_glow_iterations,
                    batch_size, optimizer_step_interval, self.model_save_dir,
                    opts.save_interval, n_trainable_params))

            self.logger.total_iterations = self.total_iterations
            self.logger.start_timer()

        output_manager = ModelOutputManager(optimizer_step_interval)

        data_loader = create_training_data_loader(
            dataset, max_samples_per_gpu, True, collate_fn, opts.num_cpu_workers,
            self.elapsed_iterations)

        self.interrupt_detector.start()

        sub_iter_idx = 0
        first_flag = True
        if self.amp:
            scaler = amp.GradScaler()

        for image_seqs, targets, meta_info in data_loader:
            if self.glow_iterations < self.total_glow_iterations and global_cfg.MODEL.USE_OFFSET_HEAD:
                join_flag = False
                self.glow_iterations += 1

            else:
                join_flag = True

            if self.glow_iterations != 0:
                first_flag = False

            if self.amp:
                with amp.autocast():
                    model_output = self.model(join_flag, first_flag, image_seqs.to(device=self.local_device),
                                              tensor_struct_to(targets, device=self.local_device))
                    dist_utils.synchronize()
                    if self.interrupt_detector.is_interrupted:
                        raise InterruptException()

                    optim_loss = output_manager(model_output)
                    # optim_loss.backward()

                scaler.scale(optim_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                model_output = self.model(join_flag, first_flag, image_seqs.to(device=self.local_device),
                                          tensor_struct_to(targets, device=self.local_device))
                dist_utils.synchronize()
                if self.interrupt_detector.is_interrupted:
                    raise InterruptException()

                optim_loss = output_manager(model_output)
                optim_loss.backward()
                self.optimizer.step()

            sub_iter_idx += 1
            if sub_iter_idx < optimizer_step_interval:
                continue

            sub_iter_idx = 0

            # self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.elapsed_iterations += 1

            logging_vars, _ = output_manager.reset()
            logging_vars = dist_utils.reduce_dict(logging_vars, average=True)
            logging_vars = {k: v.item() for k, v in logging_vars.items()}

            if self.is_main_process:
                add_to_summary = self.elapsed_iterations % opts.summary_interval == 0
                self.logger.add_training_point(self.elapsed_iterations, add_to_summary, **logging_vars)

                if hasattr(self.lr_scheduler, "get_last_lr"):  # PyTorch versions > 1.5
                    logging_vars['lr'] = self.lr_scheduler.get_last_lr()[0]
                else:
                    logging_vars['lr'] = self.lr_scheduler.get_lr()[0]

                if self.elapsed_iterations % opts.display_interval == 0:
                    log_func = self.console_logger.info
                else:
                    log_func = self.console_logger.debug

                eta, avg_time_per_iter = self.logger.compute_eta(as_string=True)
                log_func(
                    "It: {:05d} - {:s} - ETA: {:s} - sec/it: {:.3f}".format(
                        self.elapsed_iterations,
                        var_keys_to_str(logging_vars),
                        eta,
                        avg_time_per_iter))

            if self.elapsed_iterations % opts.save_interval == 0:
                if self.is_main_process:
                    # remove outdated checkpoints
                    checkpoints = sorted(glob(os.path.join(self.model_save_dir, '%06d.pth')))
                    if len(checkpoints) > opts.ckpts_to_keep:
                        for ckpt_path in checkpoints[:-opts.ckpts_to_keep]:
                            os.remove(ckpt_path)

                    self.backup_session()

                dist_utils.synchronize()

            if self.elapsed_iterations == self.total_iterations * batch_size:
                if self.is_main_process:
                    # remove outdated checkpoints
                    checkpoints = sorted(glob(os.path.join(self.model_save_dir, '%06d.pth')))
                    if len(checkpoints) > opts.ckpts_to_keep:
                        for ckpt_path in checkpoints[:-opts.ckpts_to_keep]:
                            os.remove(ckpt_path)

                    self.backup_session()

                dist_utils.synchronize()



        self.console_logger.info(
            "Training complete\n"
            "Model(s) saved to: %s\n"
            "Log file(s) saved to: %s\n" % (self.model_save_dir, self.log_dir))


def create_logger(args):
    logger = logging.getLogger("MaskTCNNTrainLogger")
    if dist_utils.is_main_process():
        logger.setLevel(args.log_level)
    else:
        logger.setLevel(args.subprocess_log_level)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(proc_id)d] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
    extra = {"proc_id": dist_utils.get_rank()}
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

    logger = logging.LoggerAdapter(logger, extra)
    logger.propagate = False

    return logger


def setup_cfg(args, model_dir, ignore_existing_cfg):
    # if a config file has been provided, load it
    if args.cfg:
        print("[ INFO] Loading config from {}".format(args.cfg))
        global_cfg.merge_from_file(args.cfg)
        return

    if ignore_existing_cfg:
        return

    # when restoring session, load the config file already present in that directory
    if args.restore_session:
        expected_config_filepath = os.path.realpath(os.path.join(args.restore_session, os.pardir, 'config.yaml'))
        print("[ INFO] Restoring config from {}".format(expected_config_filepath))
        global_cfg.merge_from_file(expected_config_filepath)

    # if the output directory already exists and there is a config file present there, then load it
    else:
        expected_config_filepath = os.path.join(model_dir, 'config.yaml')
        if os.path.exists(expected_config_filepath):
            print("[ INFO] Restoring config from {}".format(expected_config_filepath))
            global_cfg.merge_from_file(expected_config_filepath)


def start(args, cfg):
    # suppress Python warnings from sub-processes to prevent duplicate warnings being printed to console
    if dist_utils.get_rank() > 0:
        warnings.filterwarnings("ignore")

    logger = create_logger(args)
    model_save_dir = os.path.join(ModelPaths.checkpoint_base_dir(), cfg.TRAINING.MODE, args.model_dir)

    if dist_utils.is_main_process():
        os.makedirs(model_save_dir, exist_ok=True)

    # check if a checkpoint already exists in the model save directory. If it does, and the 'no_resume' flag is not set,
    # training should resume from the last pre-existing checkpoint.
    if args.model_init:
        model_init_dir = os.path.join(ModelPaths.init_param_dir(), cfg.TRAINING.MODE)
        existing_ckpts = sorted(glob(os.path.join(model_init_dir, "*.pth")))
    else:
        existing_ckpts = sorted(glob(os.path.join(model_save_dir, "*.pth")))
    if existing_ckpts and not args.no_resume:
        args.restore_session = existing_ckpts[-1]
        args.initial_ckpt = None  # when jobs auto-restart on the cluster, this might be set,
        # however we want to use the latest checkpoint instead

    # backup config to model directory
    if dist_utils.is_main_process():
        with open(os.path.join(model_save_dir, 'config.yaml'), 'w') as writefile:
            yaml.dump(global_cfg.d(), writefile)

    trainer = Trainer(cfg, model_save_dir, args, logger)

    try:
        trainer.start(args)
    except InterruptException as _:
        if dist_utils.is_main_process():
            print("Interrupt signal received. Saving checkpoint...")
            trainer.backup_session()
            dist_utils.synchronize()
        exit(1)
    except Exception as err:
        if dist_utils.is_main_process():
            print("Exception occurred. Saving checkpoint...")
            print(err)
            trainer.backup_session()
            if dist_utils.is_distributed():
                dist.destroy_process_group()
        raise err


def init_distributed(args, cfg, num_gpus):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port if args.master_port else '12356'

    # initialize the process group
    timeout = timedelta(0, 25)  # 25 seconds
    dist.init_process_group("nccl", rank=args.local_rank, world_size=num_gpus, timeout=timeout)

    try:
        start(args, cfg)
    except InterruptException as _:
        print("Training session was interrupted")

    dist_utils.synchronize()
    dist.destroy_process_group()


def main(args):
    if os.path.isabs(args.cfg):
        cfg_path = args.cfg
    else:
        cfg_path = os.path.join(RepoPaths.configs_dir(), args.cfg)

    print("Restoring config from: {}".format(cfg_path))
    global_cfg.merge_from_file(cfg_path)

    num_gpus = torch.cuda.device_count()

    if args.allow_multigpu and num_gpus > 1:
        init_distributed(args, global_cfg, num_gpus)
    else:
        start(args, global_cfg)


if __name__ == '__main__':
    parser = ArgumentParser()
    LogLevel = register_log_level_type(parser)

    # parser.add_argument('--model_dir', type=str, required=True)
    # parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default="glow_youtube_Win_v8_1.1")
    parser.add_argument('--model_init', type=str, default=True)
    parser.add_argument('--cfg', type=str, default="youtube_vis.yaml")

    # parser.add_argument('--restore_session', type=str, required=False)
    # parser.add_argument('--initial_ckpt', type=str, required=False)
    parser.add_argument('--restore_session', type=str, default=False)
    parser.add_argument('--initial_ckpt', type=str, default=False)

    parser.add_argument('--no_resume', action='store_true')

    parser.add_argument('--allow_multigpu', default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--master_port', type=str, default='12356')

    parser.add_argument('--display_interval', type=int, default=5)
    parser.add_argument('--summary_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--num_cpu_workers', type=int, default=8)
    parser.add_argument('--ckpts_to_keep', type=int, default=2)

    parser.add_argument('--log_level',            type=LogLevel, default=logging.INFO)
    parser.add_argument('--subprocess_log_level', type=LogLevel, default=logging.WARN)

    args = parser.parse_args()

    main(args)
