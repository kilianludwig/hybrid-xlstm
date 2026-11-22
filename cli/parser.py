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
import importlib
import os
import sys

_COMMAND_MODULES = [
    "cli.commands.train",
    "cli.commands.info",
    "cli.commands.benchmark",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid xLSTM training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train -c config.yaml              Train with one config file
  %(prog)s train -c config.yaml -e 200       Override epochs
  %(prog)s train --suite headline -o runs    Train one study
  %(prog)s train --suite all --model hlstm   Train one model
  %(prog)s list-tasks                        Show available tasks
  %(prog)s list-models                       Show available models
  %(prog)s benchmark                         Throughput benchmark
        """,
    )

    parser.add_argument(
        "--group",
        default=None,
        metavar="NAME",
        help=(
            "Experiment group under experiments/. "
            "Defaults to $HYBRID_GROUP, then ylstm_architecture."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    for mod_name in _COMMAND_MODULES:
        mod = importlib.import_module(mod_name)
        mod.register(subparsers)

    args = parser.parse_args()

    if getattr(args, "group", None):
        os.environ["HYBRID_GROUP"] = args.group
        from ml.models import refresh_group_overrides
        refresh_group_overrides(args.group)

    if args.command is None:
        parser.print_help()
        return

    args.func(args)
