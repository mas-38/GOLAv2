import numpy as np
import torch

from trackit.miscellanies.torch.distributed import get_rank
from trackit.core.runtime.application.builder import build_application
from trackit.core.runtime.global_constant import get_global_constant

from ...workarounds.reproducibility import seed_all_rng, enable_deterministic_computation


def prepare_application(runtime_vars, config, wandb_instance):
    performance = config['run'].get('performance', {})
    if 'torch_num_threads' in performance:
        torch.set_num_threads(performance['torch_num_threads'])
    if 'opencv_num_threads' in performance:
        import cv2
        cv2.setNumThreads(performance['opencv_num_threads'])
    global_synchronized_rng = np.random.Generator(np.random.PCG64(runtime_vars.seed))
    runtime_vars.seed = runtime_vars.seed + get_rank()
    local_rng = np.random.Generator(np.random.PCG64(runtime_vars.seed))
    instance_specific_rng = np.random.Generator(np.random.PCG64(runtime_vars.instance_id))

    seed_all_rng(runtime_vars.seed)

    # Zekai Shao: Enable deterministic computation
    enable_deterministic_computation()

    return build_application(config, runtime_vars, global_synchronized_rng, local_rng, instance_specific_rng, wandb_instance)
