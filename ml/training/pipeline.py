# Copyright 2026 Kilian Ludwig
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

import torch.nn.functional as F

from ml.models import MODEL_REGISTRY, AutoregressiveWrapper, TokenClassifier
from ml.tasks import (
    MQARConfig,
    create_mqar_dataset,
    TaskType,
    TASK_REGISTRY,
    get_task_info,
    get_task_class,
    create_task_dataset,
)


def _instantiate_model(
    model_name: str,
    input_dim: int,
    default_hidden: int,
    model_kwargs: Dict[str, Any],
):
    cls = MODEL_REGISTRY[model_name]
    params = inspect.signature(cls.__init__).parameters
    kwargs = dict(model_kwargs)
    
    if "input_dim" in params:
        kwargs.setdefault("input_dim", input_dim)
    if "hidden_dim" in params:
        kwargs.setdefault("hidden_dim", default_hidden)
    if "d_model" in params:
        kwargs.setdefault("d_model", default_hidden)
    
    if "use_positional_encoding" in params and "use_positional_encoding" not in kwargs:
        kwargs["use_positional_encoding"] = True
    if "use_cls_token" in params and "use_cls_token" not in kwargs:
        kwargs["use_cls_token"] = False
    if "max_seq_len" in params and "max_seq_len" not in kwargs:
        kwargs["max_seq_len"] = 5000
    
    return cls(**kwargs)


ModelBuilder = Callable[[], nn.Module]

_mqar_ce = nn.CrossEntropyLoss(ignore_index=-100)


def _mqar_loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    seq_len = targets.size(1)
    if outputs.size(1) > seq_len:
        outputs = outputs[:, :seq_len, :]
    elif outputs.size(1) < seq_len:
        pad_size = seq_len - outputs.size(1)
        outputs = F.pad(outputs, (0, 0, 0, pad_size))
    return _mqar_ce(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))


def _build_mqar_family(
    payload: Dict[str, Any],
    task_name: str,
    config_cls: type,
    create_dataset_fn: Callable,
    config_defaults: Optional[Dict[str, Any]] = None,
    extra_config_fields: Optional[Dict[str, Any]] = None,
) -> Tuple[Dataset, ModelBuilder, Callable, Dict, Dict]:
    task_kwargs = payload.get("task_kwargs", {})
    model_kwargs = dict(payload.get("model_kwargs", {}))
    default_hidden = model_kwargs.pop("hidden_dim", 128)

    defaults = {
        "vocab_size": 256,
        "num_kv_pairs": 8,
        "input_seq_len": 64,
        "power_a": 0.01,
        "random_non_queries": True,
    }
    if config_defaults:
        defaults.update(config_defaults)

    cfg_kwargs = {k: task_kwargs.get(k, v) for k, v in defaults.items()}
    if extra_config_fields:
        cfg_kwargs.update(extra_config_fields)
    config = config_cls(**cfg_kwargs)

    full_dataset = create_dataset_fn(
        count=payload["dataset_size"],
        config=config,
        seed=task_kwargs.get("seed", 42),
    )

    input_dim = full_dataset.input_size
    output_size = full_dataset.output_size

    def model_builder() -> TokenClassifier:
        backbone = _instantiate_model(
            payload["model"], input_dim, default_hidden, model_kwargs,
        )
        return TokenClassifier(backbone, num_classes=output_size)

    metadata = {
        "task": task_name,
        "vocab_size": config.vocab_size,
        "num_kv_pairs": config.num_kv_pairs,
        "hidden_dim": default_hidden,
        "input_dim": input_dim,
        "input_seq_len": config.input_seq_len,
    }
    if hasattr(config, "num_updates"):
        metadata["num_updates"] = config.num_updates

    return full_dataset, model_builder, _mqar_loss_fn, metadata, {
        "num_classes": output_size,
        "vocab_size": config.vocab_size,
        "num_kv_pairs": config.num_kv_pairs,
    }


def _build_mqar_components(
    payload: Dict[str, Any],
) -> Tuple[Dataset, ModelBuilder, Callable, Dict, Dict]:
    return _build_mqar_family(payload, "mqar", MQARConfig, create_mqar_dataset)


def _build_chomsky_components(
    payload: Dict[str, Any], task_name: str
) -> Tuple[Dataset, ModelBuilder, Callable, Dict, Dict]:
    task_kwargs = payload.get("task_kwargs", {})
    model_kwargs = dict(payload.get("model_kwargs", {}))
    default_hidden = model_kwargs.pop("hidden_dim", 128)

    task_info = get_task_info(task_name)
    config_class = task_info["config"]

    config_kwargs = {
        "sequence_length": task_kwargs.get("sequence_length", 32),
        "seed": task_kwargs.get("seed", 42),
    }
    for key, value in task_kwargs.items():
        if hasattr(config_class, key):
            config_kwargs[key] = value
    config = config_class(**config_kwargs)

    full_dataset = create_task_dataset(
        task_name=task_name,
        count=payload["dataset_size"],
        config=config,
    )

    input_dim = full_dataset.input_size
    output_length = full_dataset.output_len

    output_classes_spec = task_info["output_classes"]
    if callable(output_classes_spec):
        num_classes = output_classes_spec(config)
    else:
        num_classes = output_classes_spec

    def model_builder() -> AutoregressiveWrapper:
        backbone = _instantiate_model(
            payload["model"], input_dim, default_hidden, model_kwargs,
        )
        return AutoregressiveWrapper(
            backbone,
            num_classes=num_classes,
            max_output_len=output_length,
            input_dim=input_dim,
        )

    def chomsky_loss_fn(
        outputs: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        return F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )

    task_type = full_dataset.task_type
    metadata = {
        "task": task_name,
        "task_type": task_type,
        "hidden_dim": default_hidden,
        "input_dim": input_dim,
        "sequence_length": config.sequence_length,
        "output_length": output_length,
    }
    return full_dataset, model_builder, chomsky_loss_fn, metadata, {
        "num_classes": num_classes,
    }


COMPONENT_BUILDERS = {
    "mqar": _build_mqar_components,
}

for task_name in TASK_REGISTRY.keys():
    COMPONENT_BUILDERS[task_name] = (
        lambda p, tn=task_name: _build_chomsky_components(p, tn)
    )


def build_experiment_components(task_name: str, payload: Dict[str, Any]):
    if task_name not in COMPONENT_BUILDERS:
        raise ValueError(
            f"Unknown task: {task_name}. Available: {list(COMPONENT_BUILDERS.keys())}"
        )
    return COMPONENT_BUILDERS[task_name](payload)


def get_available_tasks() -> Dict[str, Dict[str, Any]]:
    tasks = {
        "mqar": {
            "name": "Standard MQAR",
            "type": "token_classification",
            "category": "mqar",
            "description": "Basic key-value associative recall",
        },
    }
    
    chomsky_levels = {
        "regular": "Type-3 (Regular)",
        "dcf": "Type-2 (DCF)",
        "cs": "Type-1 (CS)",
    }
    
    for task_name, info in TASK_REGISTRY.items():
        try:
            task_cls = get_task_class(task_name)
            task_instance = task_cls()
            output_len = task_instance.output_length(10)
            detected_type = (
                TaskType.SINGLE_LABEL if output_len == 1 else TaskType.SEQUENCE_OUTPUT
            )
        except (ImportError, TypeError, ValueError):
            detected_type = "unknown"

        tasks[task_name] = {
            "name": task_name.replace("_", " ").title(),
            "type": detected_type,
            "wrapper": "autoregressive",
            "category": "chomsky",
            "chomsky_level": chomsky_levels.get(info["module"], "Unknown"),
            "config_class": info["config"].__name__,
        }
    
    return tasks
