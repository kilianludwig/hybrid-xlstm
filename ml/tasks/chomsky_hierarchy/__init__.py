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

"""
Chomsky hierarchy task generators (originally from DeepMind).

This module provides task generators organized by the Chomsky hierarchy:
- Type-3: Regular Languages (Parity Check, Even Pairs, Cycle Navigation, Modular Arithmetic)
- Type-2: DCF Languages (Modular Arithmetic Brackets, Reverse String, Solve Equation, Stack Manipulation)  
- Type-1: CS Languages (Binary Addition/Multiplication, Bucket Sort, Compute Sqrt, Duplicate String, etc.)

All tasks are implemented in JAX and wrapped for PyTorch compatibility via adapter.py.

Task Type Detection
===================

Task types are **dynamically detected** from each task's `output_length()` method,
following the original DeepMind approach:

- output_length == 1: Single-label
- output_length > 1: Sequence-output

This aligns with the paper "Neural Networks and the Chomsky Hierarchy" (Delétang et al., ICLR 2023)
where the authors use `single_output = task.output_length(10) == 1` to determine task type.
"""

# Import the base task module (for GeneralizationTask base class)
from . import task

# Import subdirectory modules
from . import regular
from . import dcf
from . import cs

# Import task classes from subdirectories
from .regular import (
    ParityCheck,
    EvenPairs,
    CycleNavigation,
    ModularArithmetic,
)

from .dcf import (
    ModularArithmeticBrackets,
    ReverseString,
    SolveEquation,
    StackManipulation,
)

from .cs import (
    BinaryAddition,
    BinaryMultiplication,
    BucketSort,
    ComputeSqrt,
    DuplicateString,
    MissingDuplicateString,
    OddsFirst,
)

# Import adapter and configuration classes
from .adapter import (
    # Base configuration
    TaskConfig,
    
    # Task type utilities (dynamic detection)
    TaskType,
    get_task_type,
    is_single_output,
    
    # Dataset wrapper
    ChomskyTaskDataset,
    
    # Registry and factories
    TASK_REGISTRY,
    get_task_info,
    get_task_class,
    create_task_dataset,
    create_classification_dataset,
    create_seq2seq_dataset,
    
    # Regular task configs
    ParityCheckConfig,
    EvenPairsConfig,
    CycleNavigationConfig,
    ModularArithmeticConfig,
    
    # DCF task configs
    ModularArithmeticBracketsConfig,
    ReverseStringConfig,
    SolveEquationConfig,
    StackManipulationConfig,
    
    # CS task configs
    BinaryAdditionConfig,
    BinaryMultiplicationConfig,
    BucketSortConfig,
    ComputeSqrtConfig,
    DuplicateStringConfig,
    MissingDuplicateStringConfig,
    OddsFirstConfig,
    
    # Generator functions
    generate_parity_check_dataset,
    generate_even_pairs_dataset,
    generate_cycle_navigation_dataset,
    generate_modular_arithmetic_dataset,
    generate_modular_arithmetic_brackets_dataset,
    generate_reverse_string_dataset,
    generate_solve_equation_dataset,
    generate_stack_manipulation_dataset,
    generate_binary_addition_dataset,
    generate_binary_multiplication_dataset,
    generate_bucket_sort_dataset,
    generate_compute_sqrt_dataset,
    generate_duplicate_string_dataset,
    generate_missing_duplicate_string_dataset,
    generate_odds_first_dataset,
)

__all__ = [
    # Modules
    "task",
    "regular",
    "dcf", 
    "cs",
    
    # ==========================================================================
    # Base Utilities
    # ==========================================================================
    "TaskConfig",
    "TaskType",
    "get_task_type",        # New: dynamic type detection
    "is_single_output",     # New: convenience function
    "ChomskyTaskDataset",
    "TASK_REGISTRY",
    "get_task_info",
    "get_task_class",
    "create_task_dataset",
    "create_classification_dataset",
    "create_seq2seq_dataset",
    
    # ==========================================================================
    # Type-3: Regular Languages (expected: SINGLE_LABEL)
    # ==========================================================================
    # Task classes
    "ParityCheck",
    "EvenPairs",
    "CycleNavigation",
    "ModularArithmetic",
    # Task configs
    "ParityCheckConfig",
    "EvenPairsConfig",
    "CycleNavigationConfig",
    "ModularArithmeticConfig",
    
    # ==========================================================================
    # Type-2: DCF Languages (expected: varies)
    # ==========================================================================
    # Task classes
    "ModularArithmeticBrackets",  # Expected: SINGLE_LABEL
    "ReverseString",              # Expected: SEQUENCE_OUTPUT
    "SolveEquation",              # Expected: SINGLE_LABEL
    "StackManipulation",          # Expected: SEQUENCE_OUTPUT
    # Task configs
    "ModularArithmeticBracketsConfig",
    "ReverseStringConfig",
    "SolveEquationConfig",
    "StackManipulationConfig",
    
    # ==========================================================================
    # Type-1: CS Languages (expected: mostly SEQUENCE_OUTPUT)
    # ==========================================================================
    # Task classes
    "BinaryAddition",             # Expected: SEQUENCE_OUTPUT
    "BinaryMultiplication",       # Expected: SEQUENCE_OUTPUT
    "BucketSort",                 # Expected: SEQUENCE_OUTPUT
    "ComputeSqrt",                # Expected: SEQUENCE_OUTPUT
    "DuplicateString",            # Expected: SEQUENCE_OUTPUT
    "MissingDuplicateString",     # Expected: SINGLE_LABEL
    "OddsFirst",                  # Expected: SEQUENCE_OUTPUT
    # Task configs
    "BinaryAdditionConfig",
    "BinaryMultiplicationConfig",
    "BucketSortConfig",
    "ComputeSqrtConfig",
    "DuplicateStringConfig",
    "MissingDuplicateStringConfig",
    "OddsFirstConfig",
    
    # ==========================================================================
    # Generator Functions
    # ==========================================================================
    "generate_parity_check_dataset",
    "generate_even_pairs_dataset",
    "generate_cycle_navigation_dataset",
    "generate_modular_arithmetic_dataset",
    "generate_modular_arithmetic_brackets_dataset",
    "generate_reverse_string_dataset",
    "generate_solve_equation_dataset",
    "generate_stack_manipulation_dataset",
    "generate_binary_addition_dataset",
    "generate_binary_multiplication_dataset",
    "generate_bucket_sort_dataset",
    "generate_compute_sqrt_dataset",
    "generate_duplicate_string_dataset",
    "generate_missing_duplicate_string_dataset",
    "generate_odds_first_dataset",
]
