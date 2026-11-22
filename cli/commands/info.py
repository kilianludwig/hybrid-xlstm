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

import argparse
import sys


def register(subparsers) -> None:
    tasks_parser = subparsers.add_parser("list-tasks", help="List available tasks")
    tasks_parser.set_defaults(func=list_tasks)

    models_parser = subparsers.add_parser("list-models", help="List available models")
    models_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output machine-readable JSON with available and unavailable models.",
    )
    models_parser.set_defaults(func=list_models)

    info_parser = subparsers.add_parser("info", help="Show task details")
    info_parser.add_argument("task_name", help="Name of the task")
    info_parser.set_defaults(func=task_info)


def list_tasks(args: argparse.Namespace) -> None:
    from seqarch.views import show_tasks_table
    show_tasks_table()


def list_models(args: argparse.Namespace) -> None:
    import json as json_mod

    from configs.registry import MODELS
    from ml.models import MODEL_REGISTRY, XLSTM_LOAD_ERROR
    from utils.env import describe_python_env

    if getattr(args, "json_output", False):
        from ml.models import (
            MAMBA_SSM_AVAILABLE,
            MAMBAPY_AVAILABLE,
            XLSTM_AVAILABLE,
        )

        unavailable = {}
        for m in MODELS:
            if m not in MODEL_REGISTRY:
                if m == "mamba_ssm" and not MAMBA_SSM_AVAILABLE:
                    unavailable[m] = "mamba-ssm not installed (requires CUDA, Triton, causal-conv1d)"
                elif m == "mamba_py" and not MAMBAPY_AVAILABLE:
                    unavailable[m] = "mambapy not installed"
                elif m in ("slstm", "mlstm", "xlstm", "xlstm_7_1") and not XLSTM_AVAILABLE:
                    err = str(XLSTM_LOAD_ERROR).split("\n")[0][:120] if XLSTM_LOAD_ERROR else "xlstm not installed"
                    unavailable[m] = err
                else:
                    unavailable[m] = "optional dependency not installed or failed to load"

        print(json_mod.dumps({
            "available": list(MODEL_REGISTRY.keys()),
            "unavailable": unavailable,
            "env": describe_python_env(),
        }))
        return

    from seqarch.views import show_models_table
    show_models_table()


def task_info(args: argparse.Namespace) -> None:
    from seqarch.views import show_task_info
    show_task_info(args.task_name)
