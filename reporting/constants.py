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

_MAMBA_VARIANTS = {"mamba_py", "mamba_ssm"}

_MODEL_DISPLAY_ORDER = [
    "gru", "lstm", "mamba", "transformer", "mlstm", "slstm", "xlstm",
    "hlstm", "yhlstm", "xlstm_hb", "ylstm_hb",
    "ylstm_hb_c0", "ylstm_hb_c8", "ylstm_hb_c4", "ylstm_hb_c2", "ylstm_hb_c1",
    "ylstm",
]
_MODEL_DISPLAY_NAMES = {
    "lstm": "LSTM",
    "gru": "GRU",
    "mamba": "Mamba",
    "transformer": "Transformer",
    "mlstm": "mLSTM",
    "slstm": "sLSTM",
    "xlstm": "xLSTM",
    "hlstm": "HLSTM",
    "yhlstm": "yHLSTM",
    "xlstm_hb": "xLSTM-hb",
    "ylstm_hb": "yLSTM-hb",
    "ylstm": "yLSTM",
}
