from init_model import initialize_model
import sys
import time
from typing import Dict, Any
import numpy as np

def calculate_metrics(start_time: float, end_time: float, response: str, prompt: str) -> Dict[str, Any]:
    total_time = end_time - start_time
    response_tokens = len(response.split())
    prompt_tokens = len(prompt.split())
    total_tokens = response_tokens + prompt_tokens
    
    metrics = {
        "total_time_seconds": round(total_time, 3),
        "tokens_per_second": round(response_tokens / total_time, 2),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "response_length_chars": len(response),
        "memory_usage_mb": None  # Will be filled by GPU memory usage
    }
    return metrics

def get_response(llm, prompt: str) -> Dict[str, Any]:
    try:
        # Start timing
        start_time = time.time()
        
        # Generate response
        response = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        # End timing
        end_time = time.time()
        
        # Extract response text
        response_text = response['choices'][0]['message']['content']
        
        # Calculate metrics
        metrics = calculate_metrics(start_time, end_time, response_text, prompt)
        
        # Get GPU memory usage if available
        try:
            gpu_memory = llm.get_gpu_memory_usage()
            metrics["memory_usage_mb"] = round(gpu_memory / (1024 * 1024), 2)
        except:
            pass
        
        return {
            "success": True,
            "response": response_text,
            "metrics": metrics
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "metrics": None
        }

def print_metrics(result: Dict[str, Any]):
    if not result["success"]:
        print(f"\nError: {result['error']}")
        return
    
    print("\nResponse:", result["response"])
    print("\nPerformance Metrics:")
    print("-" * 50)
    metrics = result["metrics"]
    print(f"Total Time: {metrics['total_time_seconds']:.3f} seconds")
    print(f"Tokens Per Second: {metrics['tokens_per_second']:.2f}")
    print(f"Prompt Tokens: {metrics['prompt_tokens']}")
    print(f"Response Tokens: {metrics['response_tokens']}")
    print(f"Total Tokens: {metrics['total_tokens']}")
    print(f"Response Length: {metrics['response_length_chars']} characters")
    if metrics['memory_usage_mb']:
        print(f"GPU Memory Usage: {metrics['memory_usage_mb']} MB")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_response.py 'your prompt here'")
        sys.exit(1)

    model = initialize_model()
    if not model:
        print("Failed to initialize model")
        sys.exit(1)

    prompt = sys.argv[1]
    result = get_response(model, prompt)
    print_metrics(result)
