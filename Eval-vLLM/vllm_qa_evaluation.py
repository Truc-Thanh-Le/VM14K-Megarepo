#!/usr/bin/env python3
"""
VLLM Medical QA Evaluation Script with Cross-Language Analysis

This script evaluates language models on medical question-answering benchmarks
using vLLM for efficient batch inference. Includes comprehensive analysis,
visualization, batch token performance metrics, and cross-language adjustments.
"""

import os
import sys
import time
import json
import random
import warnings
import logging
import argparse
import gc
import re
import shutil
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VLLM Medical QA Evaluation with Cross-Language Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the JSONL dataset file"
    )
    parser.add_argument(
        "--cross_dataset_path",
        type=str,
        default=None,
        help="Path to cross-language dataset for adjustment factor calculation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (auto-generated if not specified)"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=10000,
        help="Number of questions to evaluate (0 for all)"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=250,
        help="Maximum tokens per response"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dataset_language",
        type=str,
        default="english",
        choices=["english", "vietnamese"],
        help="Dataset language for naming and adjustment calculations"
    )
    parser.add_argument(
        "--enable_topic_batch_analysis",
        action="store_true",
        help="Enable topic-based batch token analysis"
    )
    parser.add_argument(
        "--enable_difficulty_batch_analysis",
        action="store_true",
        help="Enable difficulty-based batch token analysis"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    
    return parser.parse_args()


def setup_environment():
    """Setup environment variables and check CUDA availability."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    try:
        shutil.rmtree("/tmp/triton_cache")
    except:
        pass
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def shuffle_options_for_sample(sample_data: Dict) -> Tuple[Dict, str, int]:
    """Shuffles options while maintaining correct answer mapping."""
    options_data = sample_data['options']
    answer_index = sample_data['answer_index']

    if isinstance(options_data, list):
        option_keys = [chr(ord('A') + i) for i in range(len(options_data))]
        options_dict = dict(zip(option_keys, options_data))
    else:
        options_dict = {k.upper(): v for k, v in options_data.items()}

    valid_letters = list(options_dict.keys())
    if isinstance(answer_index, int) and 0 <= answer_index < len(valid_letters):
        correct_answer_letter = valid_letters[answer_index]
    else:
        correct_answer_letter = str(answer_index).upper()

    correct_answer_text = options_dict[correct_answer_letter]
    option_values = list(options_dict.values())
    random.shuffle(option_values)

    shuffled_options = {}
    new_correct_letter = None

    for i, value in enumerate(option_values):
        letter = chr(ord('A') + i)
        shuffled_options[letter] = value
        if value == correct_answer_text:
            new_correct_letter = letter

    new_answer_index = ord(new_correct_letter) - ord('A')
    return shuffled_options, new_correct_letter, new_answer_index


def load_dataset(file_path: str, subset_size: int = 10000, seed: int = 42) -> List[Dict]:
    """Load and shuffle dataset from JSONL file."""
    random.seed(seed)
    dataset = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            required_fields = ['question', 'options', 'answer_index', 'medical_topic', 'difficulty_level']
            if not all(k in data for k in required_fields):
                continue

            question_text = data['question']
            options_data = data['options']
            answer_index = data['answer_index']

            try:
                shuffled_options, new_correct_letter, new_answer_index = shuffle_options_for_sample(data)
                
                dataset.append({
                    'question': question_text,
                    'options': shuffled_options,
                    'answer': new_correct_letter,
                    'medical_topic': data['medical_topic'],
                    'difficulty_level': data['difficulty_level']
                })
            except Exception:
                continue

    if subset_size and len(dataset) > subset_size:
        random.seed(seed)
        dataset = random.sample(dataset, subset_size)
    
    return dataset


def normalize_sample(data: Dict) -> Optional[Dict]:
    """Normalize a sample to consistent format."""
    if not all(field in data for field in ['question', 'options', 'answer_index', 'medical_topic', 'difficulty_level']):
        return None

    question_text = data['question']
    options_data = data['options']
    answer_index = data['answer_index']

    if isinstance(options_data, list):
        option_keys = [chr(ord('A') + i) for i in range(len(options_data))]
        options_dict = dict(zip(option_keys, options_data))
    else:
        options_dict = {k.upper(): v for k, v in options_data.items()}

    valid_letters = list(options_dict.keys())
    if isinstance(answer_index, int) and 0 <= answer_index < len(valid_letters):
        correct_answer_letter = valid_letters[answer_index]
    else:
        correct_answer_letter = str(answer_index).upper()

    return {
        "question": question_text,
        "options": options_dict,
        "answer": correct_answer_letter,
        "medical_topic": data['medical_topic'],
        "difficulty_level": data['difficulty_level']
    }


def process_medical_topics(topic_data):
    """Process medical topics from various formats."""
    if isinstance(topic_data, list):
        return [t.strip() for t in topic_data if t and t.strip()]
    elif isinstance(topic_data, str):
        return [t.strip() for t in topic_data.split(',') if t and t.strip()]
    else:
        return ['Unlabeled']


class StratifiedLanguageAdjuster:
    """
    Calculates and applies stratified language adjustment factors for cross-language 
    performance comparison between English and Vietnamese datasets.
    """
    
    def __init__(self):
        self.topic_factors = {}
        self.difficulty_factors = {}
        self.topic_difficulty_factors = {}
        self.base_language_factor = 1.0

    def calculate_stratified_factors(self, english_dataset, vietnamese_dataset, tokenizer):
        """Calculate adjustment factors stratified by topic and difficulty."""
        def _proc(topic_data):
            if isinstance(topic_data, list):
                return [t.strip() for t in topic_data if t and t.strip()]
            elif isinstance(topic_data, str):
                return [t.strip() for t in topic_data.split(',') if t and t.strip()]
            else:
                return ['Unlabeled']

        all_topics = set()
        all_difficulties = set()

        for dataset in [english_dataset, vietnamese_dataset]:
            for sample in dataset:
                topics = _proc(sample['medical_topic'])
                all_topics.update(topics)
                all_difficulties.add(sample['difficulty_level'].strip().capitalize())

        # Calculate topic-level factors
        for topic in all_topics:
            if topic and topic.strip():
                self.topic_factors[topic] = self._calculate_topic_factor(
                    english_dataset, vietnamese_dataset, topic, tokenizer
                )

        # Calculate difficulty-level factors
        for difficulty in all_difficulties:
            self.difficulty_factors[difficulty] = self._calculate_difficulty_factor(
                english_dataset, vietnamese_dataset, difficulty, tokenizer
            )

        # Calculate topic-difficulty combined factors
        for topic in all_topics:
            if topic and topic.strip():
                self.topic_difficulty_factors[topic] = {}
                for difficulty in all_difficulties:
                    self.topic_difficulty_factors[topic][difficulty] = self._calculate_topic_difficulty_factor(
                        english_dataset, vietnamese_dataset, topic, difficulty, tokenizer
                    )

    def _calculate_topic_factor(self, english_dataset, vietnamese_dataset, topic, tokenizer):
        """Calculate adjustment factor for a specific topic."""
        def _proc(topic_data):
            if isinstance(topic_data, list):
                return [t.strip() for t in topic_data if t and t.strip()]
            elif isinstance(topic_data, str):
                return [t.strip() for t in topic_data.split(',') if t and t.strip()]
            else:
                return ['Unlabeled']

        english_topic_samples = [s for s in english_dataset if topic in _proc(s['medical_topic'])]
        vietnamese_topic_samples = [s for s in vietnamese_dataset if topic in _proc(s['medical_topic'])]

        if len(english_topic_samples) < 5 or len(vietnamese_topic_samples) < 5:
            return self._get_default_factor()

        english_metrics = self._calculate_language_metrics(english_topic_samples, tokenizer)
        vietnamese_metrics = self._calculate_language_metrics(vietnamese_topic_samples, tokenizer)

        if english_metrics['avg_tokens'] == 0:
            return self._get_default_factor()

        return {
            'topic': topic,
            'token_ratio': vietnamese_metrics['avg_tokens'] / english_metrics['avg_tokens'],
            'question_length_ratio': (
                vietnamese_metrics['avg_question_length'] / english_metrics['avg_question_length']
                if english_metrics['avg_question_length'] > 0 else 1.0
            ),
            'combined_factor': self._calculate_combined_factor(english_metrics, vietnamese_metrics),
            'sample_sizes': {'english': len(english_topic_samples), 'vietnamese': len(vietnamese_topic_samples)}
        }

    def _calculate_difficulty_factor(self, english_dataset, vietnamese_dataset, difficulty, tokenizer):
        """Calculate adjustment factor for a specific difficulty level."""
        english_diff_samples = [s for s in english_dataset if s['difficulty_level'].strip().capitalize() == difficulty]
        vietnamese_diff_samples = [s for s in vietnamese_dataset if s['difficulty_level'].strip().capitalize() == difficulty]

        if len(english_diff_samples) < 5 or len(vietnamese_diff_samples) < 5:
            return self._get_default_factor()

        english_metrics = self._calculate_language_metrics(english_diff_samples, tokenizer)
        vietnamese_metrics = self._calculate_language_metrics(vietnamese_diff_samples, tokenizer)

        return {
            'difficulty': difficulty,
            'token_ratio': (vietnamese_metrics['avg_tokens'] / english_metrics['avg_tokens']) if english_metrics['avg_tokens'] > 0 else 1.0,
            'complexity_factor': self._calculate_difficulty_complexity_factor(difficulty),
            'combined_factor': self._calculate_combined_factor(english_metrics, vietnamese_metrics),
            'sample_sizes': {'english': len(english_diff_samples), 'vietnamese': len(vietnamese_diff_samples)}
        }

    def _calculate_topic_difficulty_factor(self, english_dataset, vietnamese_dataset, topic, difficulty, tokenizer):
        """Calculate adjustment factor for specific topic-difficulty combination."""
        def _proc(topic_data):
            if isinstance(topic_data, list):
                return [t.strip() for t in topic_data if t and t.strip()]
            elif isinstance(topic_data, str):
                return [t.strip() for t in topic_data.split(',') if t and t.strip()]
            else:
                return ['Unlabeled']

        english_samples = [s for s in english_dataset if (topic in _proc(s['medical_topic']) and s['difficulty_level'].strip().capitalize() == difficulty)]
        vietnamese_samples = [s for s in vietnamese_dataset if (topic in _proc(s['medical_topic']) and s['difficulty_level'].strip().capitalize() == difficulty)]

        if len(english_samples) < 3 or len(vietnamese_samples) < 3:
            return self._get_fallback_factor(topic, difficulty)

        english_metrics = self._calculate_language_metrics(english_samples, tokenizer)
        vietnamese_metrics = self._calculate_language_metrics(vietnamese_samples, tokenizer)

        return {
            'topic': topic,
            'difficulty': difficulty,
            'token_ratio': (vietnamese_metrics['avg_tokens'] / english_metrics['avg_tokens']) if english_metrics['avg_tokens'] > 0 else 1.0,
            'combined_factor': self._calculate_combined_factor(english_metrics, vietnamese_metrics),
            'confidence': min(len(english_samples), len(vietnamese_samples)) / 10.0,
            'sample_sizes': {'english': len(english_samples), 'vietnamese': len(vietnamese_samples)}
        }

    def _calculate_language_metrics(self, samples, tokenizer):
        """Calculate average token and character metrics for a set of samples."""
        total_tokens = 0
        total_question_length = 0
        for sample in samples:
            total_tokens += len(tokenizer.encode(sample['question'], add_special_tokens=False))
            total_question_length += len(sample['question'])
        count = len(samples)
        return {
            'avg_tokens': total_tokens / count if count > 0 else 0,
            'avg_question_length': total_question_length / count if count > 0 else 0
        }

    def _calculate_combined_factor(self, english_metrics, vietnamese_metrics):
        """Calculate combined adjustment factor from token and length ratios."""
        if english_metrics['avg_tokens'] == 0:
            return 1.0
        token_ratio = vietnamese_metrics['avg_tokens'] / english_metrics['avg_tokens']
        length_ratio = (
            vietnamese_metrics['avg_question_length'] / english_metrics['avg_question_length']
            if english_metrics['avg_question_length'] > 0 else 1.0
        )
        return (token_ratio + length_ratio) / 2

    def _calculate_difficulty_complexity_factor(self, difficulty):
        """Get complexity scaling factor for difficulty level."""
        complexity_map = {'Easy': 0.8, 'Moderate': 1.0, 'Hard': 1.2, 'Challenging': 1.4}
        return complexity_map.get(difficulty, 1.0)

    def _get_default_factor(self):
        """Return default factor when insufficient data."""
        return {'token_ratio': 1.0, 'combined_factor': 1.0, 'sample_sizes': {'english': 0, 'vietnamese': 0}}

    def _get_fallback_factor(self, topic, difficulty):
        """Get fallback factor using topic and difficulty averages."""
        topic_factor = self.topic_factors.get(topic, self._get_default_factor())
        difficulty_factor = self.difficulty_factors.get(difficulty, self._get_default_factor())
        return {
            'topic': topic,
            'difficulty': difficulty,
            'token_ratio': (topic_factor['token_ratio'] + difficulty_factor['token_ratio']) / 2,
            'combined_factor': (topic_factor['combined_factor'] + difficulty_factor['combined_factor']) / 2,
            'confidence': 0.5,
            'sample_sizes': {'english': 0, 'vietnamese': 0}
        }

    def get_adjustment_factor(self, topic, difficulty):
        """Get best available adjustment factor for topic-difficulty pair."""
        if topic in self.topic_difficulty_factors and difficulty in self.topic_difficulty_factors[topic]:
            return self.topic_difficulty_factors[topic][difficulty]
        elif topic in self.topic_factors:
            return self.topic_factors[topic]
        elif difficulty in self.difficulty_factors:
            return self.difficulty_factors[difficulty]
        else:
            return self._get_default_factor()


def apply_language_adjustments(performance_data, topic, difficulty, language_adjuster, dataset_language):
    """Apply language adjustment factors to performance metrics."""
    if language_adjuster is None or dataset_language == "english":
        return performance_data
    if not topic or not difficulty:
        return performance_data

    adjustment_factor = language_adjuster.get_adjustment_factor(topic, difficulty)
    factor = adjustment_factor['combined_factor']

    adjusted_data = performance_data.copy()
    adjusted_data['language_adjusted_tokens_per_second'] = performance_data.get('tokens_per_second', 0) * factor
    adjusted_data['language_adjusted_questions_per_second'] = performance_data.get('questions_per_second', 0) / factor
    adjusted_data['adjustment_factor'] = factor
    adjusted_data['adjustment_confidence'] = adjustment_factor.get('confidence', 1.0)

    return adjusted_data


def format_question(question: str, options: Dict) -> str:
    """Format question and options."""
    options_str = "\n".join([f"{key}. {value}" for key, value in sorted(options.items())])
    return f"{question}\n\n{options_str}\n\nAnswer:"


def construct_chat_prompt(question: str, options: Dict, tokenizer: Any) -> str:
    """Construct chat prompt with system message."""
    formatted_question = format_question(question, options)
    messages = [
        {"role": "system", "content": "You are a medical expert."},
        {"role": "user", "content": formatted_question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def parse_predicted_answer(response: str) -> Optional[str]:
    """Extract answer letter from model response using multiple strategies."""
    response = response.upper().strip()
    response = response.replace('*', '').replace('#', '')

    exact_patterns = [
        r'^\s*ANSWER\s*:\s*([A-Z])\s*$',
        r'^\s*THE\s+ANSWER\s+IS\s*:\s*([A-Z])\s*$',
        r'^\s*THE\s+CORRECT\s+ANSWER\s+IS\s*:\s*([A-Z])\s*$',
        r'^\s*([A-Z])\s*$',
    ]

    for pattern in exact_patterns:
        match = re.match(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    if len(response) == 1 and response.isalpha():
        return response

    line_patterns = [
        r'^\s*ANSWER\s*:\s*([A-Z])',
        r'^\s*THE\s+ANSWER\s+IS\s*:?\s*([A-Z])',
        r'^\s*CORRECT\s+ANSWER\s*:?\s*([A-Z])',
        r'^\s*([A-Z])\.',
        r'\b([A-Z])\s*IS\s+(?:THE\s+)?CORRECT',
        r'\bOPTION\s+([A-Z])\b',
        r'\b([A-Z])\s*:\s*',
    ]

    lines = response.split('\n')
    for line in lines[:5]:
        line = line.strip()
        for pattern in line_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                candidate = match.group(1).upper()
                if candidate in 'ABCDEFGH':
                    return candidate

    fallback_pattern = r'\b([A-Z])\b'
    matches = re.findall(fallback_pattern, response[:200])
    valid_matches = [m for m in matches if m in 'ABCDEFGH']

    if valid_matches:
        return valid_matches[0]

    return None


def run_inference(
    model: LLM,
    tokenizer: Any,
    dataset: List[Dict],
    sampling_params: SamplingParams
) -> Tuple[List, List, float, int, int]:
    """Run batch inference on dataset."""
    prompts = [construct_chat_prompt(sample['question'], sample['options'], tokenizer) 
               for sample in dataset]
    
    start_time = time.time()
    outputs = model.generate(prompts, sampling_params)
    end_time = time.time()
    
    generation_time = end_time - start_time
    total_tokens_generated = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    
    return outputs, prompts, generation_time, total_tokens_generated, total_prompt_tokens


def analyze_results(
    dataset: List[Dict],
    outputs: List,
    generation_time: float,
    total_tokens_generated: int,
    total_prompt_tokens: int
) -> Dict:
    """Analyze evaluation results and compute statistics."""
    total_count = 0
    correct_count = 0
    incorrect_count = 0
    deviation_count = 0
    
    topic_performance = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0, "deviations": 0})
    difficulty_performance = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0, "deviations": 0})
    topic_difficulty_performance = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "incorrect": 0, "deviations": 0}))
    
    topic_token_data = defaultdict(lambda: {"total_prompt_tokens": 0, "total_generated_tokens": 0, "count": 0})
    difficulty_token_data = defaultdict(lambda: {"total_prompt_tokens": 0, "total_generated_tokens": 0, "count": 0})
    
    generated_texts = [output.outputs[0].text for output in outputs]
    detailed_results = []
    
    for idx, (sample, output, generated_text) in enumerate(zip(dataset, outputs, generated_texts)):
        question = sample['question']
        options = sample['options']
        correct_answer = sample['answer']
        medical_topic = sample.get('medical_topic', 'Unknown')
        difficulty_level = sample.get('difficulty_level', 'Unknown')
        
        predicted_answer = parse_predicted_answer(generated_text)
        
        is_correct = (predicted_answer == correct_answer) if predicted_answer else False
        is_deviation = (predicted_answer is None)
        
        total_count += 1
        if is_correct:
            correct_count += 1
            topic_performance[medical_topic]["correct"] += 1
            difficulty_performance[difficulty_level]["correct"] += 1
            topic_difficulty_performance[medical_topic][difficulty_level]["correct"] += 1
        else:
            incorrect_count += 1
            topic_performance[medical_topic]["incorrect"] += 1
            difficulty_performance[difficulty_level]["incorrect"] += 1
            topic_difficulty_performance[medical_topic][difficulty_level]["incorrect"] += 1
        
        if is_deviation:
            deviation_count += 1
            topic_performance[medical_topic]["deviations"] += 1
            difficulty_performance[difficulty_level]["deviations"] += 1
            topic_difficulty_performance[medical_topic][difficulty_level]["deviations"] += 1
        
        topic_performance[medical_topic]["total"] += 1
        difficulty_performance[difficulty_level]["total"] += 1
        
        prompt_tokens = len(output.prompt_token_ids)
        generated_tokens = len(output.outputs[0].token_ids)
        
        topic_token_data[medical_topic]["total_prompt_tokens"] += prompt_tokens
        topic_token_data[medical_topic]["total_generated_tokens"] += generated_tokens
        topic_token_data[medical_topic]["count"] += 1
        
        difficulty_token_data[difficulty_level]["total_prompt_tokens"] += prompt_tokens
        difficulty_token_data[difficulty_level]["total_generated_tokens"] += generated_tokens
        difficulty_token_data[difficulty_level]["count"] += 1
        
        detailed_results.append({
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'is_deviation': is_deviation,
            'medical_topic': medical_topic,
            'difficulty_level': difficulty_level,
            'generated_text': generated_text,
            'prompt_tokens': prompt_tokens,
            'generated_tokens': generated_tokens
        })
    
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    average_prompt_tokens = total_prompt_tokens / total_count if total_count > 0 else 0
    average_generated_tokens = total_tokens_generated / total_count if total_count > 0 else 0
    overall_tokens_per_second = total_tokens_generated / generation_time if generation_time > 0 else 0
    
    return {
        'total_count': total_count,
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
        'deviation_count': deviation_count,
        'final_accuracy': final_accuracy,
        'generation_time': generation_time,
        'total_tokens_generated': total_tokens_generated,
        'total_prompt_tokens': total_prompt_tokens,
        'average_prompt_tokens': average_prompt_tokens,
        'average_generated_tokens': average_generated_tokens,
        'overall_tokens_per_second': overall_tokens_per_second,
        'topic_performance': dict(topic_performance),
        'difficulty_performance': dict(difficulty_performance),
        'topic_difficulty_performance': dict(topic_difficulty_performance),
        'topic_token_data': dict(topic_token_data),
        'difficulty_token_data': dict(difficulty_token_data),
        'detailed_results': detailed_results
    }


def run_batch_analysis(
    model: LLM,
    dataset: List[Dict],
    prompts: List[str],
    sampling_params: SamplingParams,
    analysis_type: str = 'topic'
) -> Dict:
    """Run batch token performance analysis by topic or difficulty."""
    results = {}
    questions_by_category = defaultdict(list)
    
    key = 'medical_topic' if analysis_type == 'topic' else 'difficulty_level'
    
    for idx, sample in enumerate(dataset):
        category = sample[key]
        questions_by_category[category].append(idx)
    
    for category, indices in tqdm(questions_by_category.items(), desc=f"Batch analysis by {analysis_type}"):
        category_prompts = [prompts[i] for i in indices]
        
        start_time = time.time()
        category_outputs = model.generate(category_prompts, sampling_params)
        end_time = time.time()
        
        category_generation_time = end_time - start_time
        category_total_tokens = sum(len(output.outputs[0].token_ids) for output in category_outputs)
        category_tokens_per_second = category_total_tokens / category_generation_time if category_generation_time > 0 else 0
        
        results[category] = {
            "batch_tokens_per_second": category_tokens_per_second,
            "total_tokens": category_total_tokens,
            "generation_time": category_generation_time,
            "num_questions": len(indices)
        }
    
    return results


def save_language_factors(language_adjuster, output_dir, model_name, dataset_name, 
                          english_size, vietnamese_size, dataset_language):
    """Save language adjustment factors to file."""
    factors_file = os.path.join(output_dir, "language_adjustment_factors.txt")
    
    with open(factors_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write("=" * 60 + "\n")
        f.write("LANGUAGE ADJUSTMENT FACTORS ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"English Dataset Size: {english_size}\n")
        f.write(f"Vietnamese Dataset Size: {vietnamese_size}\n")
        f.write(f"Current Dataset Language: {dataset_language.title()}\n\n")
        
        f.write("TOPIC-LEVEL FACTORS:\n")
        f.write("-" * 30 + "\n")
        for topic, factor in language_adjuster.topic_factors.items():
            f.write(f"\nTopic: {topic}\n")
            f.write(f"  Token Ratio (VI/ENG): {factor['token_ratio']:.3f}\n")
            f.write(f"  Combined Factor: {factor['combined_factor']:.3f}\n")
            f.write(f"  Sample Sizes - ENG: {factor['sample_sizes']['english']}, VI: {factor['sample_sizes']['vietnamese']}\n")
        
        f.write("\n\nDIFFICULTY-LEVEL FACTORS:\n")
        f.write("-" * 30 + "\n")
        for difficulty, factor in language_adjuster.difficulty_factors.items():
            f.write(f"\nDifficulty: {difficulty}\n")
            f.write(f"  Token Ratio (VI/ENG): {factor['token_ratio']:.3f}\n")
            f.write(f"  Combined Factor: {factor['combined_factor']:.3f}\n")
            f.write(f"  Sample Sizes - ENG: {factor['sample_sizes']['english']}, VI: {factor['sample_sizes']['vietnamese']}\n")


def save_text_outputs(analysis: Dict, output_dir: str, model_name: str, dataset_name: str, 
                     language_adjuster=None):
    """Save all text-based analysis outputs with language adjustments."""
    dataset_language = analysis.get('dataset_language', 'english')
    
    # Save detailed model outputs
    output_file = os.path.join(output_dir, "model_output.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, result in enumerate(analysis['detailed_results'], 1):
            f.write(f"Question {idx}:\n")
            f.write(f"Topic: {result['medical_topic']}\n")
            f.write(f"Difficulty: {result['difficulty_level']}\n")
            f.write(f"{format_question(result['question'], result['options'])}\n")
            f.write(f"Correct Answer: {result['correct_answer']}\n")
            f.write(f"Predicted Answer: {result['predicted_answer'] if result['predicted_answer'] else 'NONE (Deviation)'}\n")
            f.write(f"Result: {'CORRECT' if result['is_correct'] else 'INCORRECT'}\n")
            f.write(f"Deviation: {'YES' if result['is_deviation'] else 'NO'}\n")
            f.write(f"\nModel Output:\n{result['generated_text']}\n")
            f.write(f"\nPrompt Tokens: {result['prompt_tokens']}\n")
            f.write(f"Generated Tokens: {result['generated_tokens']}\n")
            f.write("=" * 80 + "\n\n")
    
    # Prepare topic and difficulty results with language adjustments
    topic_results = []
    for topic, perf in analysis['topic_performance'].items():
        total = perf['total']
        if total > 0:
            accuracy = (perf['correct'] / total) * 100
            
            topic_perf_data = {
                'tokens_per_second': analysis.get('topic_tps_results', {}).get(topic, {}).get('batch_tokens_per_second', 0),
                'questions_per_second': 1.0
            }
            
            if language_adjuster and dataset_language == "vietnamese":
                topic_perf_data = apply_language_adjustments(
                    topic_perf_data, topic, "Moderate", language_adjuster, dataset_language
                )
            
            topic_results.append({
                "topic": topic,
                "accuracy": accuracy,
                "correct": perf['correct'],
                "incorrect": perf['incorrect'],
                "total": total,
                "deviations": perf['deviations'],
                "performance": topic_perf_data
            })
    
    difficulty_results = []
    for difficulty, perf in analysis['difficulty_performance'].items():
        total = perf['total']
        if total > 0:
            accuracy = (perf['correct'] / total) * 100
            
            diff_perf_data = {
                'tokens_per_second': analysis.get('difficulty_tps_results', {}).get(difficulty, {}).get('batch_tokens_per_second', 0),
                'questions_per_second': 1.0
            }
            
            if language_adjuster and dataset_language == "vietnamese":
                diff_perf_data = apply_language_adjustments(
                    diff_perf_data, "Unlabeled", difficulty, language_adjuster, dataset_language
                )
            
            difficulty_results.append({
                "difficulty": difficulty,
                "accuracy": accuracy,
                "correct": perf['correct'],
                "incorrect": perf['incorrect'],
                "total": total,
                "deviations": perf['deviations'],
                "performance": diff_perf_data
            })
    
    topic_results_sorted = sorted(topic_results, key=lambda x: x['accuracy'], reverse=True)
    difficulty_order = ['Easy', 'Moderate', 'Hard', 'Challenging']
    difficulty_results_sorted = sorted(difficulty_results, 
                                      key=lambda x: difficulty_order.index(x['difficulty']) if x['difficulty'] in difficulty_order else 999)
    
    # Save comprehensive analysis
    infer_result_file = os.path.join(output_dir, "infer_result.txt")
    with open(infer_result_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Language: {dataset_language.title()}\n")
        f.write(f"Total Questions: {analysis['total_count']}\n")
        f.write(f"Correct Answers: {analysis['correct_count']}\n")
        f.write(f"Overall Accuracy: {analysis['final_accuracy']:.2%}\n")
        f.write(f"Deviations: {analysis['deviation_count']} ({analysis['deviation_count']/analysis['total_count']:.2%})\n")
        f.write(f"Generation Time: {analysis['generation_time']:.2f}s\n")
        f.write(f"Tokens/Second: {analysis['overall_tokens_per_second']:.2f}\n\n")
        
        # Language adjustments summary
        if language_adjuster and dataset_language == "vietnamese":
            f.write("LANGUAGE-ADJUSTED PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write("Performance metrics have been adjusted using cross-language factors\n")
            f.write("to enable fair comparison with English dataset results.\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("PERFORMANCE BY DIFFICULTY LEVEL\n")
        f.write("=" * 80 + "\n\n")
        
        for result in difficulty_results_sorted:
            f.write(f"{result['difficulty']} Level:\n")
            f.write(f"  Questions: {result['total']}\n")
            f.write(f"  Accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})\n")
            f.write(f"  Incorrect: {result['incorrect']}\n")
            f.write(f"  Deviations: {result['deviations']}\n")
            
            if language_adjuster and dataset_language == "vietnamese":
                perf = result['performance']
                f.write(f"  Language-Adjusted Performance:\n")
                f.write(f"    Adjusted Q/s: {perf.get('language_adjusted_questions_per_second', 0):.2f}\n")
                f.write(f"    Adjustment Factor: {perf.get('adjustment_factor', 1.0):.3f}\n")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("PERFORMANCE BY MEDICAL TOPIC (Ranked by Accuracy)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(topic_results_sorted, 1):
            f.write(f"Rank {i}: {result['topic']}\n")
            f.write(f"  Questions: {result['total']}\n")
            f.write(f"  Accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})\n")
            f.write(f"  Incorrect: {result['incorrect']}\n")
            f.write(f"  Deviations: {result['deviations']}\n")
            
            if language_adjuster and dataset_language == "vietnamese":
                perf = result['performance']
                f.write(f"  Language-Adjusted Performance:\n")
                f.write(f"    Adjusted Q/s: {perf.get('language_adjusted_questions_per_second', 0):.2f}\n")
                f.write(f"    Adjustment Factor: {perf.get('adjustment_factor', 1.0):.3f}\n")
            f.write("\n")
    
    # Save topic rankings by difficulty
    ranked_topics_file = os.path.join(output_dir, "topics_ranked_by_accuracy.txt")
    with open(ranked_topics_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write("=" * 80 + "\n")
        f.write("MEDICAL TOPICS RANKED BY ACCURACY WITHIN EACH DIFFICULTY LEVEL\n")
        f.write("=" * 80 + "\n\n")
        
        for difficulty in difficulty_order:
            f.write(f"\n{difficulty.upper()} DIFFICULTY - Topics Ranked by Accuracy:\n")
            f.write("-" * 60 + "\n")
            
            difficulty_topics = []
            for topic, topic_perf in analysis['topic_difficulty_performance'].items():
                if difficulty in topic_perf:
                    data = topic_perf[difficulty]
                    total = data['correct'] + data['incorrect']
                    if total > 0:
                        accuracy = (data['correct'] / total) * 100
                        
                        combo_perf = {'questions_per_second': 1.0}
                        if language_adjuster and dataset_language == "vietnamese":
                            combo_perf = apply_language_adjustments(
                                combo_perf, topic, difficulty, language_adjuster, dataset_language
                            )
                        
                        difficulty_topics.append({
                            'topic': topic,
                            'accuracy': accuracy,
                            'correct': data['correct'],
                            'total': total,
                            'incorrect': data['incorrect'],
                            'deviations': data['deviations'],
                            'performance': combo_perf
                        })
            
            if difficulty_topics:
                difficulty_topics.sort(key=lambda x: x['accuracy'], reverse=True)
                for rank, topic_data in enumerate(difficulty_topics, 1):
                    f.write(f"\nRank {rank}: {topic_data['topic']}\n")
                    f.write(f"  Accuracy: {topic_data['accuracy']:.2f}% ({topic_data['correct']}/{topic_data['total']})\n")
                    f.write(f"  Total Questions: {topic_data['total']}\n")
                    f.write(f"  Incorrect: {topic_data['incorrect']}\n")
                    f.write(f"  Deviations: {topic_data['deviations']}\n")
                    
                    if language_adjuster and dataset_language == "vietnamese":
                        perf = topic_data['performance']
                        f.write(f"  Adjustment Factor: {perf.get('adjustment_factor', 1.0):.3f}\n")
            else:
                f.write(f"\nNo data available for {difficulty} difficulty level.\n")


def create_visualizations(analysis: Dict, output_dir: str, model_name: str, dataset_name: str):
    """Create all visualization charts."""
    model_display_name = model_name.split('/')[-1] if '/' in model_name else model_name
    dataset_display_name = dataset_name.replace('.jsonl', '').replace('_', ' ').title()
    
    topic_results = []
    for topic, perf in analysis['topic_performance'].items():
        total = perf['total']
        if total > 0:
            accuracy = (perf['correct'] / total) * 100
            topic_results.append({
                'topic': topic,
                'accuracy': accuracy,
                'correct': perf['correct'],
                'incorrect': perf['incorrect'],
                'total': total,
                'deviations': perf['deviations']
            })
    
    difficulty_results = []
    for difficulty, perf in analysis['difficulty_performance'].items():
        total = perf['total']
        if total > 0:
            accuracy = (perf['correct'] / total) * 100
            difficulty_results.append({
                'difficulty': difficulty,
                'accuracy': accuracy,
                'correct': perf['correct'],
                'incorrect': perf['incorrect'],
                'total': total,
                'deviations': perf['deviations']
            })
    
    topic_results_sorted = sorted(topic_results, key=lambda x: x['accuracy'], reverse=True)
    
    # Topic Performance Chart
    if topic_results_sorted:
        fig, ax = plt.subplots(figsize=(20, 8))
        topics = [r['topic'] for r in topic_results_sorted]
        accuracies = [r['accuracy'] for r in topic_results_sorted]
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.3, len(topics)))
        
        bars = ax.bar(range(len(topics)), accuracies, color=colors)
        ax.set_xlabel('Medical Topic', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Medical Topic Performance - Accuracy', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.94, f'Model: {model_display_name} | Dataset: {dataset_display_name}',
                 ha='center', va='top', fontsize=12, style='italic', color='gray')
        ax.set_ylim(0, 115)
        ax.set_xticks(range(len(topics)))
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        for i, (bar, topic_data) in enumerate(zip(bars, topic_results_sorted)):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%\n({topic_data["correct"]}/{topic_data["total"]})',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(output_dir, "topic_performance_accuracy.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Difficulty Performance Chart
    if difficulty_results:
        fig, ax = plt.subplots(figsize=(12, 8))
        difficulties = [r['difficulty'] for r in difficulty_results]
        diff_accuracies = [r['accuracy'] for r in difficulty_results]
        diff_colors = ['#77dd77', '#fdfd96', '#ff6961', '#1e39ece5'][:len(difficulties)]
        
        bars = ax.bar(difficulties, diff_accuracies, color=diff_colors)
        ax.set_title('Accuracy by Difficulty Level', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.94, f'Model: {model_display_name} | Dataset: {dataset_display_name}',
                 ha='center', va='top', fontsize=12, style='italic', color='gray')
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_ylim(0, 115)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        for bar, result in zip(bars, difficulty_results):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%\n({result["correct"]}/{result["total"]})',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(output_dir, "difficulty_performance.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Deviation Rate Charts
    if difficulty_results:
        fig, ax = plt.subplots(figsize=(12, 8))
        difficulties = [r['difficulty'] for r in difficulty_results]
        deviation_rates = [(r['deviations'] / r['total']) * 100 if r['total'] > 0 else 0 for r in difficulty_results]
        diff_colors = ['#77dd77', '#fdfd96', '#ff6961', '#1e39ece5'][:len(difficulties)]
        
        bars = ax.bar(difficulties, deviation_rates, color=diff_colors)
        ax.set_title('Deviation/Hallucination Rate by Difficulty Level', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.94, f'Model: {model_display_name} | Dataset: {dataset_display_name}',
                 ha='center', va='top', fontsize=12, style='italic', color='gray')
        ax.set_ylabel('Deviation Rate (%)', fontsize=12)
        ax.set_xlabel('Difficulty Level', fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        for bar, result in zip(bars, difficulty_results):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%\n({result["deviations"]}/{result["total"]})',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(output_dir, "deviation_rate_by_difficulty.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Deviation by Topic
    if topic_results_sorted:
        sorted_by_deviation = sorted(topic_results,
                                     key=lambda x: (x['deviations'] / x['total']) * 100 if x['total'] > 0 else 0,
                                     reverse=True)
        
        topics = [r['topic'] for r in sorted_by_deviation]
        deviation_rates = [(r['deviations'] / r['total']) * 100 if r['total'] > 0 else 0 for r in sorted_by_deviation]
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(topics)))
        
        fig, ax = plt.subplots(figsize=(18, 10))
        bars = ax.bar(range(len(topics)), deviation_rates, color=colors)
        ax.set_xlabel('Medical Topic', fontsize=12, fontweight='bold')
        ax.set_ylabel('Deviation Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Deviation/Hallucination Rate by Medical Topic (Ranked)', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.94, f'Model: {model_display_name} | Dataset: {dataset_display_name}',
                 ha='center', va='top', fontsize=12, style='italic', color='gray')
        ax.set_xticks(range(len(topics)))
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        for i, (bar, result) in enumerate(zip(bars, sorted_by_deviation)):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%\n({result["deviations"]}/{result["total"]})',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(output_dir, "deviation_rate_by_topic.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Batch TPS visualizations
    if 'topic_tps_results' in analysis:
        sorted_topics_by_batch_tps = sorted(analysis['topic_tps_results'].items(),
                                          key=lambda x: x[1]['batch_tokens_per_second'], reverse=True)
        topics = [item[0] for item in sorted_topics_by_batch_tps]
        batch_tps = [item[1]['batch_tokens_per_second'] for item in sorted_topics_by_batch_tps]
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.3, len(topics)))
        
        fig, ax = plt.subplots(figsize=(18, 10))
        bars = ax.bar(range(len(topics)), batch_tps, color=colors)
        ax.set_xlabel('Medical Topic', fontsize=12, fontweight='bold')
        ax.set_ylabel('Batch Tokens/Second', fontsize=12, fontweight='bold')
        ax.set_title('Batch Token Performance by Medical Topic (Ranked)', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.94, f'Model: {model_display_name} | Dataset: {dataset_display_name}',
                 ha='center', va='top', fontsize=12, style='italic', color='gray')
        ax.set_xticks(range(len(topics)))
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(output_dir, "batch_tokens_per_second_by_topics.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    if 'difficulty_tps_results' in analysis:
        difficulties = list(analysis['difficulty_tps_results'].keys())
        batch_tps_diff = [analysis['difficulty_tps_results'][diff]['batch_tokens_per_second'] for diff in difficulties]
        diff_colors = ['#77dd77', '#fdfd96', '#ff6961', '#1e39ece5'][:len(difficulties)]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(difficulties, batch_tps_diff, color=diff_colors)
        ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Batch Tokens/Second', fontsize=12, fontweight='bold')
        ax.set_title('Batch Token Performance by Difficulty Level', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.94, f'Model: {model_display_name} | Dataset: {dataset_display_name}',
                 ha='center', va='top', fontsize=12, style='italic', color='gray')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(output_dir, "batch_tokens_per_second_by_difficulty.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Performance Dashboard
    if topic_results and difficulty_results:
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.5, f'{analysis["final_accuracy"]:.1%}', ha='center', va='center',
                fontsize=48, fontweight='bold', color='darkgreen')
        ax1.text(0.5, 0.2, f'Overall Accuracy\n({analysis["correct_count"]}/{analysis["total_count"]})',
                ha='center', va='center', fontsize=14)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        deviation_rate_percent = (analysis['deviation_count'] / analysis['total_count']) * 100 if analysis['total_count'] > 0 else 0
        ax2.text(0.5, 0.5, f'{deviation_rate_percent:.1f}%', ha='center', va='center',
                fontsize=48, fontweight='bold', color='darkred')
        ax2.text(0.5, 0.2, f'Deviation Rate\n({analysis["deviation_count"]}/{analysis["total_count"]})',
                ha='center', va='center', fontsize=14)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(0.5, 0.5, f'{analysis["overall_tokens_per_second"]:.1f}',
                ha='center', va='center', fontsize=48, fontweight='bold', color='darkblue')
        ax3.text(0.5, 0.2, 'Tokens/Second', ha='center', va='center', fontsize=14)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, :2])
        ax4.text(0.5, 0.5, f'{analysis["total_count"]}', ha='center', va='center',
                fontsize=48, fontweight='bold', color='darkblue')
        ax4.text(0.5, 0.2, 'Total Questions\nEvaluated',
                ha='center', va='center', fontsize=14)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 2])
        difficulties = [r['difficulty'] for r in difficulty_results]
        diff_accuracies = [r['accuracy'] for r in difficulty_results]
        diff_colors = ['#77dd77', '#fdfd96', '#ff6961', '#1e39ece5'][:len(difficulties)]
        
        bars = ax5.bar(difficulties, diff_accuracies, color=diff_colors)
        ax5.set_title('Difficulty Performance', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Accuracy (%)')
        ax5.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax6 = fig.add_subplot(gs[2, :])
        if 'difficulty_tps_results' in analysis:
            difficulties_tps = list(analysis['difficulty_tps_results'].keys())
            tps_values = [analysis['difficulty_tps_results'][d]['batch_tokens_per_second'] for d in difficulties_tps]
            bars = ax6.bar(difficulties_tps, tps_values, color=diff_colors[:len(difficulties_tps)])
            ax6.set_title('Token Generation Speed by Difficulty', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Tokens/Second')
            ax6.set_xlabel('Difficulty Level')
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        else:
            ax6.text(0.5, 0.5, 'Difficulty Batch Analysis Disabled', ha='center', va='center',
                    fontsize=16, style='italic', color='gray')
            ax6.axis('off')
        
        fig.suptitle(f'Performance Dashboard - {model_display_name} on {dataset_display_name}',
                    fontsize=18, fontweight='bold', y=0.98)
        plt.savefig(os.path.join(output_dir, "performance_dashboard.png"), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function."""
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = setup_environment()
    
    if args.output_dir is None:
        base_output_dir = os.path.dirname(args.dataset_path)
        dataset_filename = os.path.splitext(os.path.basename(args.dataset_path))[0]
        model_short_name = args.model_name.replace('/', '-')
        args.output_dir = os.path.join(base_output_dir, f"{model_short_name}_{dataset_filename}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    
    print("Loading main dataset...")
    dataset = load_dataset(args.dataset_path, args.subset_size, args.seed)
    print(f"Loaded {len(dataset)} samples")
    
    # Load cross-language dataset and calculate adjustment factors
    language_adjuster = None
    if args.cross_dataset_path and os.path.exists(args.cross_dataset_path):
        print("Loading cross-language dataset for adjustment factor calculation...")
        cross_dataset_raw = []
        with open(args.cross_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if all(field in data for field in ['question', 'medical_topic', 'difficulty_level', 'options', 'answer_index']):
                    normalized_sample = normalize_sample(data)
                    if normalized_sample is not None:
                        cross_dataset_raw.append(normalized_sample)
        
        print(f"Loaded {len(cross_dataset_raw)} cross-language samples")
        
        if len(cross_dataset_raw) > 100:
            print("Calculating stratified language adjustment factors...")
            language_adjuster = StratifiedLanguageAdjuster()
            
            if args.dataset_language == "english":
                language_adjuster.calculate_stratified_factors(dataset, cross_dataset_raw, tokenizer)
                save_language_factors(language_adjuster, args.output_dir, args.model_name, 
                                    os.path.basename(args.dataset_path), 
                                    len(dataset), len(cross_dataset_raw), args.dataset_language)
            elif args.dataset_language == "vietnamese":
                language_adjuster.calculate_stratified_factors(cross_dataset_raw, dataset, tokenizer)
                save_language_factors(language_adjuster, args.output_dir, args.model_name,
                                    os.path.basename(args.dataset_path),
                                    len(cross_dataset_raw), len(dataset), args.dataset_language)
            
            print("Language adjustment factors calculated and saved!")
    
    print("\nRunning main inference...")
    outputs, prompts, generation_time, total_tokens_generated, total_prompt_tokens = run_inference(
        model, tokenizer, dataset, sampling_params
    )
    
    print("Analyzing results...")
    analysis = analyze_results(
        dataset, outputs, generation_time, total_tokens_generated, total_prompt_tokens
    )
    
    analysis['dataset_language'] = args.dataset_language
    
    if args.enable_topic_batch_analysis:
        print("Running topic-based batch analysis...")
        topic_tps_results = run_batch_analysis(
            model, dataset, prompts, sampling_params, analysis_type='topic'
        )
        analysis['topic_tps_results'] = topic_tps_results
    
    if args.enable_difficulty_batch_analysis:
        print("Running difficulty-based batch analysis...")
        difficulty_tps_results = run_batch_analysis(
            model, dataset, prompts, sampling_params, analysis_type='difficulty'
        )
        analysis['difficulty_tps_results'] = difficulty_tps_results
    
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    
    print("Saving results...")
    save_text_outputs(analysis, args.output_dir, args.model_name, dataset_name, language_adjuster)
    
    print("Creating visualizations...")
    create_visualizations(analysis, args.output_dir, args.model_name, dataset_name)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Overall Accuracy: {analysis['final_accuracy']:.2%}")
    print(f"Total Questions: {analysis['total_count']}")
    print(f"Generation Time: {analysis['generation_time']:.2f}s")
    print(f"Tokens/Second: {analysis['overall_tokens_per_second']:.2f}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
