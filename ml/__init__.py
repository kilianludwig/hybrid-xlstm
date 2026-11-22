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

__version__ = "1.0.0"

from .models import (
    MODEL_REGISTRY,
    BaseSequenceModel,
    LSTMSequenceModel,
    GRUSequenceModel,
    TransformerSequenceModel,
    AutoregressiveWrapper,
    TokenClassifier,
)

from .tasks import (
    SequenceTask,
    TaskType,
    TaskConfig,
    TASK_REGISTRY,
    create_task_dataset,
    get_task_info,
)

from .training.config import (
    ExperimentConfig,
    WandbConfig,
)

from .training.trainer import (
    Trainer,
    TrainerConfig,
    TrainingListener,
)

from .training.output import (
    RunOutput,
    OutputConfig,
    generate_run_name,
)

from utils import set_determinism, ParameterStats, count_parameters

__all__ = [
    "__version__",
    
    "MODEL_REGISTRY",
    "BaseSequenceModel",
    "LSTMSequenceModel",
    "GRUSequenceModel",
    "TransformerSequenceModel",
    "AutoregressiveWrapper",
    "TokenClassifier",
    
    "SequenceTask",
    "TaskType",
    "TaskConfig",
    "TASK_REGISTRY",
    "create_task_dataset",
    "get_task_info",
    
    "ExperimentConfig",
    "WandbConfig",
    "Trainer",
    "TrainerConfig",
    "TrainingListener",
    
    "RunOutput",
    "OutputConfig",
    "generate_run_name",

    "set_determinism",
    "ParameterStats",
    "count_parameters",
]
