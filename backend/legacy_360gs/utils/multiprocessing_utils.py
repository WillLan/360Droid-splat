import copy
import io

import torch
import torch.multiprocessing as mp
from torch import nn


GAUSSIAN_CPU_ONLY_ATTRS = {
    "unique_kfIDs",
    "n_obs",
    "_is_sky",
    "_layer",
    "_layer_pending",
    "_region_tag",
    "_anchor_kf",
    "_anchor_submap",
    "_birth_frame",
    "_anchor_kf_pending",
    "_region_tag_pending",
    "_anchor_submap_pending",
    "_birth_frame_pending",
    "_anchor_level",
    "_anchor_voxel_size",
    "_is_sky_anchor",
    "_anchor_grid_coord",
    "_anchor_obs_count",
    "_anchor_last_seen_kf",
    "_anchor_conf_accum",
    "_anchor_level_pending",
    "_anchor_voxel_pending",
    "_is_sky_anchor_pending",
    "_anchor_grid_coord_pending",
    "_anchor_obs_count_pending",
    "_anchor_last_seen_kf_pending",
    "_anchor_conf_accum_pending",
}

QUEUE_MESSAGE_BYTES_TAG = "__s3po_queue_message_bytes_v1__"


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        raise mp.queues.Empty

    def qsize(self):
        return 0

    def empty(self):
        return True


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj


def _clone_tensor_to_device(tensor, device):
    if isinstance(tensor, nn.Parameter):
        cloned = tensor.detach().clone().to(device=device)
        return nn.Parameter(cloned, requires_grad=tensor.requires_grad)
    return tensor.detach().clone().to(device=device)


def _move_obj_to_device(obj, device, skip_attrs, cpu_only_attrs, visited):
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    if isinstance(obj, (torch.Tensor, nn.Parameter)):
        return _clone_tensor_to_device(obj, device)

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            target_device = "cpu" if key in cpu_only_attrs else device
            obj[key] = _move_obj_to_device(
                value,
                target_device,
                skip_attrs=skip_attrs,
                cpu_only_attrs=cpu_only_attrs,
                visited=visited,
            )
        return obj

    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            obj[idx] = _move_obj_to_device(
                value,
                device,
                skip_attrs=skip_attrs,
                cpu_only_attrs=cpu_only_attrs,
                visited=visited,
            )
        return obj

    if isinstance(obj, tuple):
        return tuple(
            _move_obj_to_device(
                value,
                device,
                skip_attrs=skip_attrs,
                cpu_only_attrs=cpu_only_attrs,
                visited=visited,
            )
            for value in obj
        )

    if isinstance(obj, set):
        return {
            _move_obj_to_device(
                value,
                device,
                skip_attrs=skip_attrs,
                cpu_only_attrs=cpu_only_attrs,
                visited=visited,
            )
            for value in obj
        }

    if hasattr(obj, "__dict__"):
        for attr, value in list(vars(obj).items()):
            if attr in skip_attrs:
                setattr(obj, attr, None)
                continue
            target_device = "cpu" if attr in cpu_only_attrs else device
            moved = _move_obj_to_device(
                value,
                target_device,
                skip_attrs=skip_attrs,
                cpu_only_attrs=cpu_only_attrs,
                visited=visited,
            )
            setattr(obj, attr, moved)
        return obj

    return obj


def move_obj_to_device_(obj, device="cuda", skip_attrs=None, cpu_only_attrs=None):
    skip_attrs = set(skip_attrs or ())
    cpu_only_attrs = set(cpu_only_attrs or ())
    return _move_obj_to_device(
        obj,
        device=device,
        skip_attrs=skip_attrs,
        cpu_only_attrs=cpu_only_attrs,
        visited=set(),
    )


def clone_obj_to_device(obj, device="cpu", skip_attrs=None, cpu_only_attrs=None):
    cloned = copy.deepcopy(obj)
    return move_obj_to_device_(
        cloned,
        device=device,
        skip_attrs=skip_attrs,
        cpu_only_attrs=cpu_only_attrs,
    )


def pack_queue_message(message):
    """Serialize queue payloads without PyTorch's shared-storage fd handoff.

    Directly putting tensors on a torch multiprocessing queue uses a resource
    sharer file descriptor. If the producer process exits or faults before the
    consumer finishes rebuilding storages, the consumer sees FileNotFoundError.
    Serializing to plain bytes makes the queue transfer independent of producer
    process lifetime.
    """
    buffer = io.BytesIO()
    torch.save(message, buffer)
    return (QUEUE_MESSAGE_BYTES_TAG, buffer.getvalue())


def unpack_queue_message(message, map_location="cpu"):
    if (
        isinstance(message, tuple)
        and len(message) == 2
        and message[0] == QUEUE_MESSAGE_BYTES_TAG
    ):
        buffer = io.BytesIO(message[1])
        try:
            return torch.load(buffer, map_location=map_location, weights_only=False)
        except TypeError:
            buffer.seek(0)
            return torch.load(buffer, map_location=map_location)
    return message
