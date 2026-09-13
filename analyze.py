import sys
import os
import re
import math
import statistics
from collections import Counter

# Ensure UTF-8 output even on Windows console
sys.stdout.reconfigure(encoding='utf-8')

def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def corpus_bleu(hypotheses, references, max_n=4):
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_ngrams = _ngrams(hyp, n)
            ref_ngrams = _ngrams(ref, n)
            for ng, cnt in hyp_ngrams.items():
                clipped_counts[n-1] += min(cnt, ref_ngrams.get(ng, 0))
            total_counts[n-1] += max(0, len(hyp) - n + 1)

    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        log_avg = sum(math.log(p) for p in precisions) / max_n
        bp = math.exp(min(0, 1 - ref_len / max(1, hyp_len)))
        bleu = 100.0 * bp * math.exp(log_avg)

    bp_val = math.exp(min(0, 1 - ref_len / max(1, hyp_len)))
    return bleu, precisions, bp_val, hyp_len, ref_len

def analyze_logs():
    print("==================================================================")
    print("                 TRAINING LOGS (train_logs.txt)                   ")
    print("==================================================================")
    log_path = os.path.join('checkpoints', 'train_logs.txt')
    if not os.path.exists(log_path):
        print(f"Log file not found at {log_path}")
        return
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Pre-process: split concatenated lines like "...tok/s=1,1402026-09-10..."
    lines = []
    for raw_line in text.splitlines():
        # if there are multiple timestamps in a single line, split them
        parts = re.split(r'(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', raw_line)
        for p in parts:
            p = p.strip()
            if p:
                lines.append(p)

    log_pattern = re.compile(
        r'(?P<time>[\d\-]+ [\d:,]+) \| INFO \| ep (?P<ep>\d+)/(?P<max_ep>\d+) \| step\s+(?P<step>\d+) \| loss=(?P<loss>[\d\.]+) \| lr=(?P<lr>[\d\.e\-\+]+) \| gnorm=(?P<gnorm>[\d\.]+) \| ε=(?P<eps>[\d\.]+) \| tok/s=(?P<toks>[\d,]+)'
    )

    parsed_logs = []
    for line in lines:
        m = log_pattern.search(line)
        if m:
            d = m.groupdict()
            toks_str = d['toks'].replace(',', '')
            # handle case where timestamp digits leaked into tok/s
            if len(toks_str) > 6:
                toks_str = toks_str[:4]
            parsed_logs.append({
                'time': d['time'],
                'ep': int(d['ep']),
                'step': int(d['step']),
                'loss': float(d['loss']),
                'lr': float(d['lr']),
                'gnorm': float(d['gnorm']),
                'eps': float(d['eps']),
                'toks': int(toks_str)
            })

    print(f"Total valid parsed log records: {len(parsed_logs)}")
    if not parsed_logs:
        return

    first = parsed_logs[0]
    last = parsed_logs[-1]
    min_loss_entry = min(parsed_logs, key=lambda x: x['loss'])
    max_loss_entry = max(parsed_logs, key=lambda x: x['loss'])
    
    print(f"Training Span: {first['time']} to {last['time']}")
    print(f"Step Range:    {first['step']} -> {last['step']} (across {len(parsed_logs)*100} logged steps)")
    print(f"Initial Loss:  {first['loss']:.4f} (Step {first['step']})")
    print(f"Final Loss:    {last['loss']:.4f} (Step {last['step']})")
    print(f"Global Min Loss: {min_loss_entry['loss']:.4f} at Step {min_loss_entry['step']} (Epoch {min_loss_entry['ep']})")

    # Group by epoch
    epochs = {}
    for entry in parsed_logs:
        ep = entry['ep']
        if ep not in epochs:
            epochs[ep] = []
        epochs[ep].append(entry)

    for ep, entries in sorted(epochs.items()):
        steps = [e['step'] for e in entries]
        losses = [e['loss'] for e in entries]
        toks = [e['toks'] for e in entries]
        gnorms = [e['gnorm'] for e in entries]
        print(f"\n--- Epoch {ep} Overview ({len(entries)} check-ins) ---")
        print(f"  Steps:      {min(steps)} to {max(steps)}")
        print(f"  Loss:       start={entries[0]['loss']:.4f} -> end={entries[-1]['loss']:.4f}")
        print(f"              min={min(losses):.4f}, max={max(losses):.4f}, mean={statistics.mean(losses):.4f}")
        print(f"  Gradient Norm (gnorm): mean={statistics.mean(gnorms):.3f}, max={max(gnorms):.3f}")
        print(f"  Throughput: mean={statistics.mean(toks):.1f} tok/s (median={statistics.median(toks):.1f})")

    # Trace significant speed and loss transitions
    # Trace significant speed and loss transitions dynamically
    print("\n--- Phase Transitions & Notable Events in Logs ---")
    total_len = len(parsed_logs)
    if total_len > 0:
        num_phases = min(4, total_len)
        chunk_size = total_len // num_phases
        for i in range(num_phases):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_phases - 1 else total_len
            phase_logs = parsed_logs[start_idx:end_idx]
            if phase_logs:
                mean_toks = statistics.mean([e['toks'] for e in phase_logs])
                mean_loss = statistics.mean([e['loss'] for e in phase_logs])
                start_step = phase_logs[0]['step']
                end_step = phase_logs[-1]['step']
                print(f"Phase {i+1} (Steps {start_step} - {end_step}):")
                print(f"  Mean tok/s: {mean_toks:.1f}, Mean loss: {mean_loss:.4f}")
                
                # Basic heuristic notes based on observed loss and throughput
                if i == 0:
                    print("  Note: Initial training phase. Loss typically drops rapidly.")
                elif mean_toks < statistics.mean([e['toks'] for e in parsed_logs]) * 0.8:
                    print("  Note: Throughput dropped significantly in this phase. Possible longer sequences or batching changes.")
                elif mean_loss <= min_loss_entry['loss'] * 1.05:
                    print("  Note: Model reaching lowest loss region. Gradient norms should be stabilizing.")
                print()

    # Teacher forcing schedule in logs
    epsilons = [e['eps'] for e in parsed_logs]
    print(f"\nTeacher Forcing Rate (ε):")
    print(f"  Min ε: {min(epsilons):.3f}, Max ε: {max(epsilons):.3f}")
    print(f"  ε values: {set(epsilons)}")

def analyze_eval():
    print("\n==================================================================")
    print("             EVALUATION ANALYSIS (evaluation_result.txt)          ")
    print("==================================================================")
    with open('evaluation_result.txt', 'r', encoding='utf-8', errors='ignore') as f:
        eval_text = f.read()

    ckpt_match = re.search(r'Checkpoint loaded <- (.*)', eval_text)
    if ckpt_match:
        print(f"Evaluated Checkpoint: {ckpt_match.group(1)}")

    entry_pattern = re.compile(
        r'\[\s*(?P<idx>\d+)\]\s+EN:\s*(?P<en>.*?)\n\s+HY:\s*(?P<hy>.*?)\n\s+RE:\s*(?P<re>.*?)\n\s+BLEU:\s*(?P<bleu>[\d\.]+)',
        re.DOTALL
    )

    entries = []
    for m in entry_pattern.finditer(eval_text):
        entries.append({
            'idx': int(m.group('idx')),
            'en': m.group('en').strip(),
            'hy': m.group('hy').strip(),
            're': m.group('re').strip(),
            'bleu': float(m.group('bleu'))
        })

    total_sents = len(entries)
    print(f"Total Sentences Evaluated before Interrupt: {total_sents}")
    if total_sents == 0:
        return

    # Calculate actual Corpus BLEU
    hyps = [e['hy'].split() for e in entries]
    refs = [e['re'].lower().split() for e in entries]

    c_bleu, precisions, bp, h_len, r_len = corpus_bleu(hyps, refs, max_n=4)

    print(f"\n--- ACTUAL CORPUS BLEU-4 (Across all {total_sents} sentences) ---")
    print(f"  Corpus BLEU-4:       {c_bleu:.2f} / 100.00")
    print(f"  Brevity Penalty (BP):{bp:.4f} (Hypothesis tokens: {h_len}, Reference tokens: {r_len}, Ratio: {h_len/r_len:.3f})")
    print(f"  1-gram Precision:    {precisions[0]*100:.2f}%")
    print(f"  2-gram Precision:    {precisions[1]*100:.2f}%")
    print(f"  3-gram Precision:    {precisions[2]*100:.2f}%")
    print(f"  4-gram Precision:    {precisions[3]*100:.2f}%")

    # Sentence-level stats
    bleus = [e['bleu'] for e in entries]
    print(f"\n--- Sentence-level BLEU Statistics ---")
    print(f"  Sentence Mean BLEU:   {statistics.mean(bleus):.4f}")
    print(f"  Sentence Median BLEU: {statistics.median(bleus):.4f}")
    print(f"  Sentence Max BLEU:    {max(bleus):.4f}")
    print(f"  Zero-BLEU count:      {sum(1 for b in bleus if b == 0.0)} / {total_sents} ({sum(1 for b in bleus if b == 0.0)/total_sents*100:.1f}%)")
    print(f"    (Note: Zero sentence-BLEU occurs whenever a sentence has 0 matching 4-grams, despite having valid 1/2/3-grams)")

    # Qualitative analysis: Translation Quality Buckets
    print("\n--- Sample High Quality Translations ---")
    sorted_by_bleu = sorted(entries, key=lambda x: x['bleu'], reverse=True)
    for i, e in enumerate(sorted_by_bleu[:5]):
        print(f"  [{e['idx']}] BLEU: {e['bleu']:.4f}")
        print(f"       EN: {e['en']}")
        print(f"       HY: {e['hy']}")
        print(f"       RE: {e['re']}\n")

    # Low / Zero BLEU samples that actually have decent translations
    print("--- Sample Zero-Sentence-BLEU but Good Partial Matches ---")
    zero_entries = [e for e in entries if e['bleu'] == 0.0]
    # check 1-gram precision of zero-bleu
    partially_good = []
    for e in zero_entries:
        h_words = set(e['hy'].lower().split())
        r_words = set(e['re'].lower().split())
        overlap = len(h_words & r_words) / max(1, len(h_words))
        if overlap > 0.4 and len(h_words) > 5:
            partially_good.append((overlap, e))
    partially_good.sort(key=lambda x: x[0], reverse=True)
    for overlap, e in partially_good[:3]:
        print(f"  [{e['idx']}] Word Overlap: {overlap*100:.1f}% (BLEU-4=0.0 due to no matching 4-gram)")
        print(f"       EN: {e['en']}")
        print(f"       HY: {e['hy']}")
        print(f"       RE: {e['re']}\n")

if __name__ == '__main__':
    analyze_logs()
    analyze_eval()
