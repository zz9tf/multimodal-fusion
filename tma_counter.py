import os, re, csv, numpy as np

base = "/home/zheng/zheng/mini2/hancock_data/TMA/TMA_Core_encodings"
markers = ["CD3","CD8","CD56","CD68","CD163","HE","MHC1","PDL1"]
paths = {m: os.path.join(base, f"tma_uni_tile_1024_{m}.npz") for m in markers}

# 规范化键：只取 block, x, y, patient 四要素
pat = re.compile(r"_block(\d+)_x(\d+)_y(\d+)_patient(\w+)$")

norm_sets = {}
raw_counts = {}
for m, p in paths.items():
    if not os.path.exists(p):
        print(f"[ERR] missing file for {m}: {p}")
        continue
    d = np.load(p, allow_pickle=True)
    ks = list(d.keys())
    s = set()
    for k in ks:
        m2 = pat.search(k)
        if m2:
            s.add((int(m2.group(1)), int(m2.group(2)), int(m2.group(3)), m2.group(4)))
    norm_sets[m] = s
    raw_counts[m] = len(ks)
    print(f"{m}: raw_keys={len(ks)} norm_keys={len(s)}")

if not norm_sets:
    raise SystemExit("No normalized key sets loaded")

# 交集（可被 8 个 marker 全部覆盖的样本）
inter = set.intersection(*norm_sets.values())
print(f"\nIntersection (normalized) size: {len(inter)}")

# 是否所有样本都能组合（= 是否并集与交集相等）
norm_union = set().union(*norm_sets.values())
all_combinable = len(norm_union) == len(inter)
print(f"All samples combinable (all 8 markers available): {all_combinable}")
print(f"Total normalized unique tuples: {len(norm_union)}")

# 找出不能组合的样本（缺少至少一个 marker）并列出缺失的 marker 列表
incomplete = sorted(list(norm_union - inter))
print(f"Incomplete tuples (missing at least one marker): {len(incomplete)}")

# 输出前若干条样例
for t in incomplete[:10]:
    present = [m for m in markers if t in norm_sets[m]]
    missing = [m for m in markers if t not in norm_sets[m]]
    print("missing_tuple:", t, "| missing:", missing, "| present:", present)

# 导出完整缺失明细到 CSV
report_csv = "/home/zheng/zheng/multimodal-fusion/tma_missing_report.csv"
with open(report_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["block","x","y","patient","missing_markers","present_markers"])
    for t in incomplete:
        b,x,y,pid = t
        present = [m for m in markers if t in norm_sets[m]]
        missing = [m for m in markers if t not in norm_sets[m]]
        writer.writerow([b,x,y,pid, ",".join(missing), ",".join(present)])
print(f"\nMissing report saved: {report_csv}")