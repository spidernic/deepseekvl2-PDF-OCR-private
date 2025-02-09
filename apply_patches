# =========================================================================
# script apply_patches.py
# Objective: Apply custom patches to DeepSeek-VL2 for multi-GPU support
# =========================================================================
# Author: Nicolas Cravino
# Created: February 9, 2025
# Acknowledgments:
# - https://github.com/deepseek-ai/DeepSeek-VL2.git
# - Paper Reference:
#   @misc{wu2024deepseekvl2mixtureofexpertsvisionlanguagemodels,
#         title={DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding},
#         author={Wu, Zhiyu and Chen, Xiaokang and Pan, Zizheng and Liu, Xingchao and Liu, Wen and Dai, Damai and 
#                Gao, Huazuo and Ma, Yiyang and Wu, Chengyue and Wang, Bingxuan and Xie, Zhenda and Wu, Yu and 
#                Hu, Kai and Wang, Jiawei and Sun, Yaofeng and Li, Yukun and Piao, Yishi and Guan, Kang and 
#                Liu, Aixin and Xie, Xin and You, Yuxiang and Dong, Kai and Yu, Xingkai and Zhang, Haowei and 
#                Zhao, Liang and Wang, Yisong and Ruan, Chong},
#         year={2024},
#         eprint={2412.10302},
#         archivePrefix={arXiv},
#         primaryClass={cs.CV},
#         url={https://arxiv.org/abs/2412.10302}
#   }
# 
# =========================================================================
# Copyright 2025 Nicolas Cravino
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
# =========================================================================

import os
import shutil
import sys
from pathlib import Path

def apply_patches():
    """Apply custom patches to DeepSeek-VL2 for multi-GPU support."""
    # Get the directory where this script is located
    current_dir = Path(__file__).parent.absolute()
    
    # Find DeepSeek-VL2 installation
    try:
        import deepseek_vl2
        deepseek_path = Path(deepseek_vl2.__file__).parent
        print(f"Found DeepSeek-VL2 installation at: {deepseek_path}")
    except ImportError:
        print("Error: DeepSeek-VL2 not found. Please install it first using:")
        print("git clone https://github.com/deepseek-ai/DeepSeek-VL2.git")
        print("cd DeepSeek-VL2")
        print("pip install -e .")
        sys.exit(1)

    # Backup original file
    inference_path = deepseek_path / "serve" / "inference.py"
    backup_path = inference_path.with_suffix('.py.backup')
    
    if not backup_path.exists():
        print(f"Creating backup of original inference.py at: {backup_path}")
        shutil.copy2(inference_path, backup_path)
    
    # Copy our modified version
    modified_inference = current_dir / "inference.py"
    if not modified_inference.exists():
        print("Error: Modified inference.py not found in patches directory")
        sys.exit(1)
    
    print(f"Applying multi-GPU patch to: {inference_path}")
    shutil.copy2(modified_inference, inference_path)
    print("Successfully applied multi-GPU patch!")

if __name__ == "__main__":
    apply_patches()
