# Constants for ANSI colors
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"


# === STUDENT CLUB ===
# Without join
SC_tp = [0,1,1,1,1,24,0,1,2,11,0,1,1,0,0,1,1,0,0,1,1,2,1,0,1,4,1,2,2,0,21,1,8,0]
SC_fp = [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,1,0,1,0,3,0,0,0,1,3,0,0,1]
SC_fn = [1,0,0,0,0,0,1,0,5,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1]
SC_uScore = [1,5,5,5,5,5,2,5,2,5,2,5,5,2.5,1,5,5,1,2.5,4,5,4,5,1,5,3,5,5,5,1,4.5,3,5,1]

tp = sum(SC_tp)
fp = sum(SC_fp)
fn = sum(SC_fn)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
avg_uScore = sum(SC_uScore) / len(SC_uScore)

print(f"--- {CYAN}STUDENT CLUB Performance (NO join){RESET} ---")
print(f"Precision: {GREEN}{precision:.4f}{RESET}")
print(f"Recall: {GREEN}{recall:.4f}{RESET}")
print(f"F1-score: {GREEN}{f1:.4f}{RESET}")
print(f"uScore: {GREEN}{avg_uScore:.4f}{RESET}")

# With join
SC_j_tp = [1,6,1,3,3,1,3,0,0,0]
SC_j_fp = [0,7,17,0,0,0,0,0,0,0]
SC_j_fn = [0,0,0,11,5,1,0,2,1,1]
SC_j_uScore = [5,2,2,2,2,2,5,2,2,2]

tp = sum(SC_j_tp)
fp = sum(SC_j_fp)
fn = sum(SC_j_fn)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
avg_uScore = sum(SC_j_uScore) / len(SC_j_uScore)

print(f"--- {CYAN}STUDENT CLUB Performance (join){RESET} ---")
print(f"Precision: {GREEN}{precision:.4f}{RESET}")
print(f"Recall: {GREEN}{recall:.4f}{RESET}")
print(f"F1-score: {GREEN}{f1:.4f}{RESET}")
print(f"uScore: {GREEN}{avg_uScore:.4f}{RESET}")
print("\n\n")
# ====================

# === SUPERHERO ===
# Without join
SH_tp = [1,0,1,0,0,0,1,1,0,1,8,1,1]
SH_fp = [0,1,0,1,1,1,0,0,1,0,0,0,0]
SH_fn = [0,1,0,1,1,1,0,0,1,0,1,9,0]
SH_uScore = [5,1,5,1,2,1,5,5,2,5,4.5,1,5]

tp = sum(SH_tp)
fp = sum(SH_fp)
fn = sum(SH_fn)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
avg_uScore = sum(SH_uScore) / len(SH_uScore)

print(f"--- {CYAN}SUPERHERO Performance (NO join){RESET} ---")
print(f"Precision: {GREEN}{precision:.4f}{RESET}")
print(f"Recall: {GREEN}{recall:.4f}{RESET}")
print(f"F1-score: {GREEN}{f1:.4f}{RESET}")
print(f"uScore: {GREEN}{avg_uScore:.4f}{RESET}")

# With join
SH_j_tp = [0,0,1,0,0,0,1,]
SH_j_fp = [0,1,0,1,0,0,0,]
SH_j_fn = [1,1,0,1,1,1,0,]
SH_j_uScore = [2,1,5,1,2,2,5,]

tp = sum(SH_j_tp)
fp = sum(SH_j_fp)
fn = sum(SH_j_fn)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
avg_uScore = sum(SH_j_uScore) / len(SH_j_uScore)

print(f"--- {CYAN}SUPERHERO Performance (join){RESET} ---")
print(f"Precision: {GREEN}{precision:.4f}{RESET}")
print(f"Recall: {GREEN}{recall:.4f}{RESET}")
print(f"F1-score: {GREEN}{f1:.4f}{RESET}")
print(f"uScore: {GREEN}{avg_uScore:.4f}{RESET}")
print("\n\n")
# ====================

# === THROMBOSIS PREDICTIONS ===
# Without join
TP_tp = [0,1,0,0,1,1,0,0,0,3,0,4,3,0,0,1,0,13,0,0,1,0,0]
TP_fp = [1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0]
TP_fn = [1,0,1,1,0,0,1,1,1,9,1,5,0,1,2,0,1,3,1,1,0,1,1]
TP_uScore = [1,5,1,2,5,5,2,1,2,2.5,2,2.5,5,2,2,5,1,4,1,2,5,2,2]

tp = sum(TP_tp)
fp = sum(TP_fp)
fn = sum(TP_fn)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
avg_uScore = sum(TP_uScore) / len(TP_uScore)

print(f"--- {CYAN}THROMBOSIS PREDICTIONS Performance (NO join){RESET} ---")
print(f"Precision: {GREEN}{precision:.4f}{RESET}")
print(f"Recall: {GREEN}{recall:.4f}{RESET}")
print(f"F1-score: {GREEN}{f1:.4f}{RESET}")
print(f"uScore: {GREEN}{avg_uScore:.4f}{RESET}")

# With join
TP_j_tp = [1,0,0,0]
TP_j_fp = [0,0,0,1]
TP_j_fn = [0,1,1,1]
TP_j_uScore = [5,2,2,1]

tp = sum(TP_j_tp)
fp = sum(TP_j_fp)
fn = sum(TP_j_fn)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
avg_uScore = sum(TP_j_uScore) / len(TP_j_uScore)

print(f"--- {CYAN}THROMBOSIS PREDICTIONS Performance (join){RESET} ---")
print(f"Precision: {GREEN}{precision:.4f}{RESET}")
print(f"Recall: {GREEN}{recall:.4f}{RESET}")
print(f"F1-score: {GREEN}{f1:.4f}{RESET}")
print(f"uScore: {GREEN}{avg_uScore:.4f}{RESET}")
print("\n\n")
# ====================

#--------------------AVG-----------------------

# Without join
tp = sum(SC_tp + SH_tp + TP_tp)
fp = sum(SC_fp + SH_fp + TP_fp)
fn = sum(SC_fn + SH_fn + TP_fn)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
avg_uScore = sum(SC_uScore + SH_uScore + TP_uScore) / len(SC_uScore + SH_uScore + TP_uScore)

print(f"--- {CYAN}AVG Performance (NO join){RESET} ---")
print(f"Precision: {GREEN}{precision:.4f}{RESET}")
print(f"Recall: {GREEN}{recall:.4f}{RESET}")
print(f"F1-score: {GREEN}{f1:.4f}{RESET}")
print(f"uScore: {GREEN}{avg_uScore:.4f}{RESET}")

# With join
tp = sum(SC_j_tp + SH_j_tp + TP_j_tp)
fp = sum(SC_j_fp + SH_j_fp + TP_j_fp)
fn = sum(SC_j_fn + SH_j_fn + TP_j_fn)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
avg_uScore = sum(SC_j_uScore + SH_j_uScore + TP_j_uScore) / len(SC_j_uScore + SH_j_uScore + TP_j_uScore)

print(f"--- {CYAN}AVG Performance (join){RESET} ---")
print(f"Precision: {GREEN}{precision:.4f}{RESET}")
print(f"Recall: {GREEN}{recall:.4f}{RESET}")
print(f"F1-score: {GREEN}{f1:.4f}{RESET}")
print(f"uScore: {GREEN}{avg_uScore:.4f}{RESET}")
print("\n\n")