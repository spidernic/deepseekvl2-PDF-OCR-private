# =========================================================================
# script ocr.py
# Objective: Scan PDF files and extract text using DeepSeek-VL2 that is awesome
# =========================================================================
# Author: Nicolas Cravino
# Created: February 9, 2025
# Acknowledgments:
# - https://github.com/deepseek-ai/DeepSeek-VL2.git
# - Paper Reference:
#   @misc{wu2024deepseekvl2mixtureofexpertsvisionlanguagemodels,
#         title={DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding},
#         author={Zhiyu Wu and Xiaokang Chen and Zizheng Pan and Xingchao Liu and Wen Liu and Damai Dai and Huazuo Gao 
#                and Yiyang Ma and Chengyue Wu and Bingxuan Wang and Zhenda Xie and Yu Wu and Kai Hu and Jiawei Wang 
#                and Yaofeng Sun and Yukun Li and Yishi Piao and Kang Guan and Aixin Liu and Xin Xie and Yuxiang You 
#                and Kai Dong and Xingkai Yu and Haowei Zhang and Liang Zhao and Yisong Wang and Chong Ruan},
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

# Import required libraries

import os
os.environ["XFORMERS_DISABLE_TRITON"] = "1"

import argparse
import logging
import sys
from pathlib import Path

import torch
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.serve.inference import convert_conversation_to_prompts, deepseek_generate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_markdown_content(content):
    """Clean up the markdown content by removing special tokens and normalizing whitespace."""
    # Remove special tokens and normalize whitespace
    content = content.replace("<｜end▁of▁sentence｜>", "")
    content = content.replace("<|User|>", "")
    content = content.replace("<|Assistant|>", "")
    content = content.replace("<image>", "")
    
    # Remove extra newlines
    content = "\n".join(line for line in content.splitlines() if line.strip())
    return content

def process_image(image_path, vl_gpt, vl_chat_processor, tokenizer):
    """Process a single image and extract text using DeepSeek-VL2."""
    logger.info(f"Processing image: {image_path}")
    
    try:
        # Load image
        image = Image.open(image_path)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise
    
    # Initialize conversation
    logger.info("Initializing conversation")
    try:
        conversation = vl_chat_processor.new_chat_template()
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise
    
    # Prepare prompt
    text = """<image>\n<|ref|>Please perform OCR on this image. Extract and transcribe all text content exactly as it appears, maintaining the original formatting, layout, and structure. Include all visible text, numbers, and special characters.<|/ref|>"""
    
    # Add messages to conversation
    logger.info("Adding messages to conversation")
    try:
        conversation.append_message(conversation.roles[0], (text, [image]))
        conversation.append_message(conversation.roles[1], "")
    except Exception as e:
        logger.error(f"Error adding messages to conversation: {str(e)}")
        raise
    
    # Convert conversation to prompts
    logger.info("Converting conversation to prompts")
    try:
        all_conv, last_image = convert_conversation_to_prompts(conversation)
    except Exception as e:
        logger.error(f"Error converting conversation to prompts: {str(e)}")
        raise
    
    # Process inputs
    logger.info("Processing inputs with vl_chat_processor")
    try:
        prepare_inputs = vl_chat_processor(
            conversations=all_conv,
            images=[last_image],
            inference_mode=True,
            force_batchify=True,
            system_prompt=""
        )
    except Exception as e:
        logger.error(f"Error processing inputs: {str(e)}")
        raise
    
    # Move inputs to device
    logger.info("Moving inputs to device")
    try:
        for key in prepare_inputs.__dict__:
            if isinstance(prepare_inputs.__dict__[key], torch.Tensor):
                prepare_inputs.__dict__[key] = prepare_inputs.__dict__[key].to(vl_gpt.device)
    except Exception as e:
        logger.error(f"Error moving inputs to device: {str(e)}")
        raise
    
    # Generate response
    logger.info("Generating response")
    try:
        outputs = deepseek_generate(
            conversations=all_conv,
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            tokenizer=tokenizer,
            stop_words=conversation.stop_str,
            max_length=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1,
            chunk_size=512
        )
        
        # Join response parts and clean
        content = "".join(outputs)
        return clean_markdown_content(content)
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def process_pdf(pdf_path, model_name, output_path=None, gpu_id=0):
    """Process a PDF file and extract text using DeepSeek-VL2."""
    logger.info(f"Processing PDF: {pdf_path} on GPU {gpu_id}")
    
    # Create output path if not provided
    if output_path is None:
        output_path = pdf_path.rsplit('.', 1)[0] + '.md'
    
    # Create temporary directory for images
    temp_dir = Path(f"temp_images_{gpu_id}")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Convert PDF to images
        logger.info(f"Converting PDF to images: {pdf_path}")
        images = convert_from_path(pdf_path)
        
        # Initialize model and processor
        logger.info(f"Loading model: {model_name} on GPU {gpu_id}")
        vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_name)
        tokenizer = vl_chat_processor.tokenizer
        
        # Set the GPU device
        device = f'cuda:{gpu_id}'
        vl_gpt = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()
        
        # Process each page
        all_text = []
        for i, image in enumerate(tqdm(images, desc=f"Processing pages on GPU {gpu_id}")):
            # Save image temporarily
            image_path = temp_dir / f"page_{i+1}.png"
            image.save(image_path)
            
            # Process image
            text = process_image(image_path, vl_gpt, vl_chat_processor, tokenizer)
            if text:
                all_text.append(f"## Page {i+1}\n\n{text}\n")
            
            # Clean up temporary image
            image_path.unlink()
        
        # Write output
        if all_text:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(all_text))
            logger.info(f"Output written to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path} on GPU {gpu_id}: {str(e)}")
        raise
    finally:
        # Clean up temporary directory if empty
        try:
            temp_dir.rmdir()
        except OSError:
            logger.warning(f"Could not remove temporary directory: {temp_dir}")

def process_pdfs_parallel(pdf_files, model_name, output_dir):
    """Process PDFs in parallel using multiple GPUs."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Using {num_gpus} GPUs for parallel processing")
    
    # Create a thread-safe counter for GPU assignment
    gpu_counter = threading.Lock()
    current_gpu = 0
    
    def get_next_gpu():
        nonlocal current_gpu
        with gpu_counter:
            gpu = current_gpu
            current_gpu = (current_gpu + 1) % num_gpus
            return gpu
    
    def process_pdf_with_gpu(pdf_file):
        gpu_id = get_next_gpu()
        output_path = output_dir / f"{pdf_file.stem}.md"
        try:
            process_pdf(str(pdf_file), model_name, str(output_path), gpu_id)
        except Exception as e:
            logger.error(f"Failed to process {pdf_file} on GPU {gpu_id}: {str(e)}")
    
    # Process PDFs in parallel using a thread pool
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        list(executor.map(process_pdf_with_gpu, pdf_files))

def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF using DeepSeek-VL2")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to PDF file or directory containing PDFs")
    parser.add_argument("--output_path", type=str, help="Path to output directory (optional, defaults to same location as PDFs)")
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path)
    
    if pdf_path.is_file():
        # Process single PDF file
        output_path = args.output_path if args.output_path else None
        process_pdf(str(pdf_path), args.model_name, output_path, gpu_id=0)
    
    elif pdf_path.is_dir():
        # Process all PDFs in directory
        output_dir = Path(args.output_path) if args.output_path else pdf_path
        output_dir.mkdir(exist_ok=True)
        
        # Find all PDF files in directory
        pdf_files = list(pdf_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in directory")
        
        if len(pdf_files) > 1 and torch.cuda.device_count() > 1:
            # Use parallel processing with multiple GPUs
            process_pdfs_parallel(pdf_files, args.model_name, output_dir)
        else:
            # Process sequentially on single GPU
            for pdf_file in pdf_files:
                output_path = output_dir / f"{pdf_file.stem}.md"
                try:
                    process_pdf(str(pdf_file), args.model_name, str(output_path))
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file}: {str(e)}")
                    continue
    
    else:
        logger.error(f"Path does not exist or is neither a file nor directory: {pdf_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
