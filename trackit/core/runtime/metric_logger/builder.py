from . import MetricLogger
from trackit.core.runtime.build_context import BuildContext
from .adaptors.local.builder import construct_local_metric_logger
from .adaptors.wandb.builder import construct_wandb_metric_logger
from .adaptors.jsonl import construct_jsonl_metric_logger


def build_metric_logger(logging_config, build_context: BuildContext, output_path=None, task_name=None) -> MetricLogger:
    logger_dispatcher = MetricLogger()
    construct_local_metric_logger(logger_dispatcher, logging_config, build_context)
    mirrors_wandb = construct_jsonl_metric_logger(logger_dispatcher, logging_config, build_context, output_path, task_name)
    if build_context.wandb_instance is not None and not mirrors_wandb:
        construct_wandb_metric_logger(logger_dispatcher, logging_config, build_context)
    return logger_dispatcher
