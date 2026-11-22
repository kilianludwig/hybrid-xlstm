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

"""Deterministic context-free language tasks (DCF subset of Type-2)."""

from .modular_arithmetic_brackets import ModularArithmeticBrackets
from .reverse_string import ReverseString
from .solve_equation import SolveEquation
from .stack_manipulation import StackManipulation

__all__ = [
    "ModularArithmeticBrackets",
    "ReverseString",
    "SolveEquation",
    "StackManipulation",
]

