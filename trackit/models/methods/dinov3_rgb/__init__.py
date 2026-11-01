"""RGB tracking pretraining with reusable DINOv3 tracking features."""

from .backbone import TrackingCrop, TrackingFeatureBundle
from .reference import FrozenReferenceEncoder, ReferenceContext
from .history import CausalFeatureHistory, HistoryRead
from .feature_contract import FeatureObservation, FeatureSupervision, FutureTarget, ReconstructionTarget
from .alignment import (align_target_roi, reconstruction_target, future_target,
                        crop_boxes_from_siamfc, siamfc_from_crop_boxes, masked_cosine_loss)
