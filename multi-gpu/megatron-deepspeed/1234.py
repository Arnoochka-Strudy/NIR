# Преобразует обычный txt в JSONL
with open("benchmark_mean.txt", "r", encoding="utf-8") as fin, open("benchmark_mean.jsonl", "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if line:
            fout.write(f'{{"text": "{line}"}}\n')
