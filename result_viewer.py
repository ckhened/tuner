import os
import re
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import argparse


def get_json(fn):
    j = None
    try:
        with open(fn, 'r') as f:
            j = json.load(f)
    except Exception as e:
        print(f"Opening file {fn} failed with exception {e}")
        #sys.exit(1)
    return j


def get_result_files(directory):
    all_files = defaultdict(lambda: defaultdict(list))
    for root, dirs, files in os.walk(directory):
        for file in files:
            #all_files.append(os.path.join(root, file))
            r = root[len(directory)+1:].split('/')
            all_files[r[0]][r[1]].append(os.path.join(root, file))            
            #all_files[(r[0], r[1])].append(file)
    return all_files


def get_highest_concurrency_file(files):
    t = [(int(f[f.rfind('C')+1:f.rfind('.')]), f) for f in files]
    return max(t)


def extract_results_from_json(fn):
    j = get_json(fn)
    return {
        'Model': j['model_id'],
        'Input Tokens': int(j['total_input_tokens']/j['num_prompts']),
        'Output Tokens': int(j['total_output_tokens']/j['num_prompts']),
        'Concurrency': int(j['request_rate']),
        'Output tput (tokens/sec)': round(j['output_throughput'], 2),
        'Total tput (tokens/sec)': round(j['total_token_throughput'], 2),
        'TTFT avg (ms)': round(j['mean_ttft_ms'], 2),
        'TTFT P50 (ms)': round(j['p50_ttft_ms'], 2),
        'TTFT P90 (ms)': round(j['p90_ttft_ms'], 2),
        'TTFT P99 (ms)': round(j['p99_ttft_ms'], 2),
        'TPOT avg (ms)': round(j['mean_tpot_ms'], 2),
        'TPOT P50 (ms)': round(j['p50_tpot_ms'], 2),
        'TPOT P90 (ms)': round(j['p90_tpot_ms'], 2),
        'TPOT P99 (ms)': round(j['p99_tpot_ms'], 2),
        'Request Throughput': round(j['request_throughput'], 2),
        'Request Latency avg (s)': round(j['mean_e2el_ms']/1000, 2),
        'Request Latency P50 (s)': round(j['p50_e2el_ms']/1000, 2),
        'Request Latency P90 (s)': round(j['p90_e2el_ms']/1000, 2),
        'Request Latency P99 (s)': round(j['p99_e2el_ms']/1000, 2),
    }


parser = argparse.ArgumentParser(description="Generate result csv file from tester results directory")
parser.add_argument("-d", "--result-dir", type=str, help="result dir path", required=True)
parser.add_argument("-c", "--csv-file", type=str, help="output csv file name with path", required=True)
parser.add_argument("-s", "--small-csv", type=str, help="output small csv file name with path", required=True)
args = parser.parse_args()

files = get_result_files(args.result_dir)

values = []
for batches in files.values():
    for fns in batches.values():
        _, fn = get_highest_concurrency_file(fns)
        values.append(extract_results_from_json(fn))
df = pd.DataFrame(values)
df = df.sort_values(by=['Model', 'Input Tokens'], ascending=[True, True])

df.to_csv(args.csv_file, index=False)
df[['Model', 'Input Tokens', 'Output Tokens', 'Concurrency', 'Output tput (tokens/sec)', 'TTFT P90 (ms)', 'TPOT P90 (ms)']].to_csv(args.small_csv, index=False)

