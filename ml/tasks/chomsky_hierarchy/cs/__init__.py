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

"""Context-sensitive language tasks (Type-1 in Chomsky hierarchy).

Note: These are labeled 'cs' in the original DeepMind repository but include
tasks that go beyond context-sensitive languages in computational complexity.
"""

from .binary_addition import BinaryAddition
from .binary_multiplication import BinaryMultiplication
from .bucket_sort import BucketSort
from .compute_sqrt import ComputeSqrt
from .duplicate_string import DuplicateString
from .missing_duplicate_string import MissingDuplicateString
from .odds_first import OddsFirst

__all__ = [
    "BinaryAddition",
    "BinaryMultiplication",
    "BucketSort",
    "ComputeSqrt",
    "DuplicateString",
    "MissingDuplicateString",
    "OddsFirst",
]

