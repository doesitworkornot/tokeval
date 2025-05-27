import sys
import pandas as pd
import json

def parquet_to_jsonl(parquet_path, jsonl_path):
    df = pd.read_parquet(parquet_path)
    with open(jsonl_path, 'w', encoding='utf-8') as fout:
        for record in df.to_dict(orient='records'):
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parquet_to_jsonl.py input.parquet output.jsonl")
        sys.exit(1)
    parquet_to_jsonl(sys.argv[1], sys.argv[2])