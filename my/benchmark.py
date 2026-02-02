#!/usr/bin/env python3

"""

vLLM Prefill Performance Benchmark



Tests latency vs prompt length (prefill performance) on a vLLM server.

Measures time-to-first-token (TTFT) across exponentially growing input lengths.

"""



import argparse

import time

import statistics

import csv

from datetime import datetime

from openai import OpenAI





def generate_prompt(length: int) -> str:

    """

    Generate a prompt of approximately the specified token length.

    Uses repetitive text to ensure consistent token count.

    """

    # Approximate: ~1.3 tokens per word for English text

    # Use a base phrase and repeat to reach target length

    base_phrase = "The quick brown fox jumps over the lazy dog. "

    words_needed = int(length / 1.3)

    words_in_phrase = len(base_phrase.split())

    repetitions = max(1, words_needed // words_in_phrase)



    prompt = base_phrase * repetitions

    return prompt.strip()





def measure_ttft(client: OpenAI, model: str, prompt: str, max_tokens: int = 1) -> float:

    """

    Measure Time-To-First-Token (TTFT) for a given prompt.



    Returns:

        TTFT in seconds

    """

    start_time = time.perf_counter()



    # Use streaming to measure TTFT accurately

    stream = client.chat.completions.create(

        model=model,

        messages=[{"role": "user", "content": prompt}],

        max_tokens=max_tokens,

        stream=True,

        temperature=0.0,

    )



    # Get first token

    first_token_time = None

    for chunk in stream:

        if chunk.choices[0].delta.content:

            first_token_time = time.perf_counter()

            break



    # Consume remaining stream

    for _ in stream:

        pass



    if first_token_time is None:

        first_token_time = time.perf_counter()



    ttft = first_token_time - start_time

    return ttft





def run_benchmark(

    base_url: str,

    model: str,

    min_length: int,

    max_length: int,

    num_trials: int,

    num_warmup: int,

    api_key: str = "EMPTY"

):

    """

    Run the prefill performance benchmark.

    """

    client = OpenAI(base_url=base_url, api_key=api_key)



    # Generate prompt lengths (powers of 2)

    prompt_lengths = []

    current = min_length

    while current < max_length:

        prompt_lengths.append(current)

        current *= 2



    # Always include max_length (even if not a power of 2)

    if not prompt_lengths or prompt_lengths[-1] != max_length:

        prompt_lengths.append(max_length)



    print(f"vLLM Prefill Benchmark")

    print(f"{'='*60}")

    print(f"Model: {model}")

    print(f"Server: {base_url}")

    print(f"Prompt lengths: {prompt_lengths}")

    print(f"Warmup runs: {num_warmup}")

    print(f"Trial runs per length: {num_trials}")

    print(f"{'='*60}\n")



    results = []



    for prompt_length in prompt_lengths:

        print(f"Testing prompt length: {prompt_length} tokens")

        prompt = generate_prompt(prompt_length)



        # Warmup runs

        print(f"  Running {num_warmup} warmup iterations...")

        for i in range(num_warmup):

            try:

                measure_ttft(client, model, prompt)

            except Exception as e:

                print(f"  Warmup {i+1} failed: {e}")



        # Actual benchmark runs

        print(f"  Running {num_trials} benchmark iterations...")

        ttfts = []

        for i in range(num_trials):

            try:

                ttft = measure_ttft(client, model, prompt)

                ttfts.append(ttft)

                print(f"    Trial {i+1}: TTFT = {ttft*1000:.2f} ms")

            except Exception as e:

                print(f"    Trial {i+1} failed: {e}")



        if ttfts:

            avg_ttft = statistics.mean(ttfts)

            median_ttft = statistics.median(ttfts)

            min_ttft = min(ttfts)

            max_ttft = max(ttfts)

            stddev_ttft = statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0



            print(f"  Results:")

            print(f"    Mean TTFT:   {avg_ttft*1000:.2f} ms")

            print(f"    Median TTFT: {median_ttft*1000:.2f} ms")

            print(f"    Min TTFT:    {min_ttft*1000:.2f} ms")

            print(f"    Max TTFT:    {max_ttft*1000:.2f} ms")

            print(f"    Std Dev:     {stddev_ttft*1000:.2f} ms")

            print(f"    Throughput:  {prompt_length/avg_ttft:.2f} tokens/sec")

            print()



            results.append({

                'prompt_length': prompt_length,

                'mean_ttft_ms': avg_ttft * 1000,

                'median_ttft_ms': median_ttft * 1000,

                'min_ttft_ms': min_ttft * 1000,

                'max_ttft_ms': max_ttft * 1000,

                'stddev_ttft_ms': stddev_ttft * 1000,

                'throughput_tokens_per_sec': prompt_length / avg_ttft,

                'num_trials': len(ttfts)

            })



    return results





def save_results(results: list, output_file: str, model: str, base_url: str):

    """

    Save benchmark results to CSV file.

    """

    timestamp = datetime.now().isoformat()



    with open(output_file, 'w', newline='') as f:

        writer = csv.writer(f)



        # Write metadata

        writer.writerow(['# Benchmark Metadata'])

        writer.writerow(['Timestamp', timestamp])

        writer.writerow(['Model', model])

        writer.writerow(['Server', base_url])

        writer.writerow([])



        # Write results header

        writer.writerow([

            'prompt_length',

            'mean_ttft_ms',

            'median_ttft_ms',

            'min_ttft_ms',

            'max_ttft_ms',

            'stddev_ttft_ms',

            'throughput_tokens_per_sec',

            'num_trials'

        ])



        # Write results

        for result in results:

            writer.writerow([

                result['prompt_length'],

                f"{result['mean_ttft_ms']:.2f}",

                f"{result['median_ttft_ms']:.2f}",

                f"{result['min_ttft_ms']:.2f}",

                f"{result['max_ttft_ms']:.2f}",

                f"{result['stddev_ttft_ms']:.2f}",

                f"{result['throughput_tokens_per_sec']:.2f}",

                result['num_trials']

            ])



    print(f"Results saved to: {output_file}")





def main():

    parser = argparse.ArgumentParser(

        description='Benchmark vLLM prefill performance across different prompt lengths',

        formatter_class=argparse.ArgumentDefaultsHelpFormatter

    )



    parser.add_argument(

        '--base-url',

        type=str,

        default='http://localhost:8000/v1',

        help='vLLM server base URL (OpenAI-compatible endpoint)'

    )



    parser.add_argument(

        '--model',

        type=str,

        required=True,

        help='Model name to test'

    )



    parser.add_argument(

        '--min-length',

        type=int,

        default=128,

        help='Minimum prompt length in tokens'

    )



    parser.add_argument(

        '--max-length',

        type=int,

        default=8192,

        help='Maximum prompt length in tokens'

    )



    parser.add_argument(

        '--trials',

        type=int,

        default=10,

        help='Number of benchmark trials per prompt length'

    )



    parser.add_argument(

        '--warmup',

        type=int,

        default=3,

        help='Number of warmup runs per prompt length'

    )



    parser.add_argument(

        '--output',

        type=str,

        default='vllm_prefill_results.csv',

        help='Output CSV file for results'

    )



    parser.add_argument(

        '--api-key',

        type=str,

        default='EMPTY',

        help='API key for authentication (use EMPTY for local vLLM)'

    )



    args = parser.parse_args()



    # Run benchmark

    results = run_benchmark(

        base_url=args.base_url,

        model=args.model,

        min_length=args.min_length,

        max_length=args.max_length,

        num_trials=args.trials,

        num_warmup=args.warmup,

        api_key=args.api_key

    )



    # Save results

    if results:

        save_results(results, args.output, args.model, args.base_url)



        print(f"\n{'='*60}")

        print("Benchmark Summary:")

        print(f"{'='*60}")

        print(f"{'Prompt Length':<15} {'Mean TTFT (ms)':<15} {'Throughput (tok/s)':<20}")

        print(f"{'-'*60}")

        for result in results:

            print(f"{result['prompt_length']:<15} "

                  f"{result['mean_ttft_ms']:<15.2f} "

                  f"{result['throughput_tokens_per_sec']:<20.2f}")

    else:

        print("No results collected!")





if __name__ == '__main__':

    main()
