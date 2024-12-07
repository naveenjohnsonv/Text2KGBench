import argparse
import json
import os
import sys
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def load_file(file_path: str) -> dict:
    """Load JSON config file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return {}

def load_sentences(file_path: str) -> Tuple[List[str], List[str]]:
    """Load sentences and IDs from JSONL file"""
    sentences, ids = [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                sentences.append(data['sent'])
                ids.append(data['id'])
        return sentences, ids
    except Exception as e:
        print(f"Error loading sentences from {file_path}: {str(e)}")
        return [], []

def compute_similarities(model: SentenceTransformer, 
                       test_sentences: List[str],
                       train_sentences: List[str],
                       test_ids: List[str],
                       train_ids: List[str],
                       top_k: int) -> dict:
    """Compute similarities between test and train sentences"""
    try:
        # Compute embeddings
        print('Computing embeddings for test sentences...')
        test_embeddings = model.encode(test_sentences, convert_to_tensor=True, show_progress_bar=True)
        print('Computing embeddings for train sentences...')
        train_embeddings = model.encode(train_sentences, convert_to_tensor=True, show_progress_bar=True)

        # Compute similarities and find top-k similar sentences
        similarity_results = {}
        print('Computing similarities and finding top similar sentences...')
        for idx, test_embedding in enumerate(tqdm(test_embeddings)):
            cosine_scores = util.cos_sim(test_embedding, train_embeddings)[0]
            top_results = torch.topk(cosine_scores, k=top_k)
            similar_train_ids = [train_ids[i] for i in top_results[1]]
            similarity_results[test_ids[idx]] = similar_train_ids

        return similarity_results
    except Exception as e:
        print(f"Error computing similarities: {str(e)}")
        return {}

def process_ontology(onto: str, 
                    config: dict, 
                    model: SentenceTransformer) -> None:
    """Process single ontology"""
    try:
        # Get file paths using patterns
        test_file = config['path_patterns']['test'].replace('$$onto$$', onto)
        train_file = config['path_patterns']['train'].replace('$$onto$$', onto)
        output_file = config['path_patterns']['sent_sim'].replace('$$onto$$', onto)

        print(f'Processing ontology: {onto}')
        
        # Load test and train data
        test_sentences, test_ids = load_sentences(test_file)
        train_sentences, train_ids = load_sentences(train_file)
        
        if not test_sentences or not train_sentences:
            print(f"Skipping {onto} due to missing data")
            return

        # Compute similarities
        similarity_results = compute_similarities(
            model=model,
            test_sentences=test_sentences,
            train_sentences=train_sentences,
            test_ids=test_ids,
            train_ids=train_ids,
            top_k=config.get('top_k', 5)
        )

        if similarity_results:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(similarity_results, f, indent=4)
            print(f'Results saved to {output_file}\n')

    except Exception as e:
        print(f"Error processing ontology {onto}: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_gen_config_path', required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config
    config = load_file(args.prompt_gen_config_path)
    if not config:
        sys.exit(1)

    try:
        # Initialize model
        model_name = config.get('model_name', 'sentence-t5-xxl')
        model = SentenceTransformer(model_name)

        # Process each ontology
        for onto in config['onto_list']:
            process_ontology(onto, config, model)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()