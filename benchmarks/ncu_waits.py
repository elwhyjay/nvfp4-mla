"""Where the warps of the SM100 kernel (nvfp4_mla.sm100.DecodeV2) spend their stall samples: per role, the share of
work and of waiting on each mbarrier.

  ncu --set full --import-source yes -k regex:DecodeV2 -o prof python ...      # profile one decode call
  ncu --import prof.ncu-rep --page source --csv --print-source sass > prof.sass.csv
  python benchmarks/ncu_waits.py prof.sass.csv [--stages 4] [--p-buffers 2] [--v-parts 2] [--tma]

A try_wait names its mbarrier by its offset in the shared-memory Storage struct, and a failed try_wait branches to an
out-of-line retry loop, so its samples sit on that branch: both are attributed to the barrier. Every sample goes to
the role whose code it is in; the roles are told apart by markers in the SASS (setmaxnreg, the first wait of a
role), so a few setup instructions near a boundary can land in the neighbouring role.
"""
import argparse
import csv
import re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("sass_csv")
ap.add_argument("--stages", type=int, default=4)
ap.add_argument("--p-buffers", type=int, default=2)
ap.add_argument("--v-parts", type=int, default=1)
ap.add_argument("--tma", action="store_true", help="TMA build (tma, cvt8): TMA warp and scale-layout warps")
a = ap.parse_args()

rows = list(csv.reader(open(a.sass_csv)))
hdr = rows[1]
data = rows[2:]
ix = {h: i for i, h in enumerate(hdr)}
src = [r[ix["Source"]].strip() for r in data]
smp = [float(r[ix["Warp Stall Sampling (All Samples)"]] or 0) for r in data]
reasons = [h for h in hdr if h.startswith("stall_") and "Not Issued" not in h]

# Storage: kvm [full x stages, empty x stages], kvc [...], s [full, empty], v [full x parts, empty x parts],
# p [full x buffers, empty x buffers], o (lazy), tma [stages]; each mbarrier is 8 bytes. "waiter: what it waits for"
names, off = {}, 0
for cnt, full, empty in [(a.stages, "a published tile (kvm full)", "QK done with a stage (kvm empty)"),
                         (a.stages, "a published tile (kvc full)", "convert done with a stage (kvc empty)"),
                         (1, "S ready", "S read by the softmax"),
                         (a.v_parts, "V in TMEM", "PV done with V"),
                         (a.p_buffers, "P ready", "a free P buffer")]:
    for k in range(cnt):
        names[off + 8 * k], names[off + 8 * (cnt + k)] = full, empty
    off += 16 * cnt
names[off] = "PV(i - 1) done (lazy rescale)"
for k in range(a.stages):
    names[off + 8 + 8 * k] = "the TMA landed"
TMA0 = off + 8

WAIT = re.compile(r"SYNCS\.PHASECHK\.TRANS64(?:\.TRYWAIT)? P\d, \[[^\]]*?(?:\+0x([0-9a-f]+))?\]")
wait_off = {k: int(m.group(1) or "0", 16) for k, x in enumerate(src) if (m := WAIT.search(x))}
label = {k: names.get(o, f"barrier +0x{o:x}") for k, o in wait_off.items()}
addr = {int(r[ix["Address"]], 16): k for k, r in enumerate(data)}
for k, x in enumerate(src):
    b = re.match(r"@!P\d\s+BRA 0x([0-9a-f]+)", x)
    t = addr.get(int(b.group(1), 16)) if b else None
    if t is not None:
        j = next((j for j in range(t, min(t + 3, len(src))) if j in label), None)
        if j is not None:
            label[k] = label[j]
for k, x in enumerate(src):
    if "NANOSLEEP" in x and k - 1 in label:
        label[k] = label[k - 1]


def first(pred, start=0):
    return next((k for k in range(start, len(src)) if pred(k)), None)


setmax = [k for k, x in enumerate(src) if "USETMAXREG" in x]
if not a.tma:  # roles in code order, each starting at its setmaxnreg
    bounds = list(zip([0] + setmax, ["prologue", "load", "convert", "MMA", "softmax"]))
else:  # the cp.async load block is still compiled but no warp runs it; convert has a setmaxnreg only with 32 heads
    dealloc = [k for k in setmax if "DEALLOC" in src[k]]
    smx = next(k for k in setmax if "ALLOC" in src[k] and "DEALLOC" not in src[k])
    dead, wg3 = dealloc[0], dealloc[-1]
    kv_full = lambda k: wait_off.get(k) is not None and 8 * a.stages <= wait_off[k] < 12 * a.stages
    bounds = [(0, "prologue"), (dead, "cp.async load (not run)"), (first(kv_full, dead), "convert"), (wg3, "TMA warp"),
              (first(lambda k: wait_off.get(k, -1) >= TMA0, wg3), "scale warps"),
              (first(lambda k: wait_off.get(k, -1) == 0 or "UTCQMMA" in src[k], wg3), "MMA"), (smx, "softmax")]
    bounds = sorted((k, n) for k, n in bounds if k is not None)

tail = max((k for k, x in enumerate(src) if "EXIT" in x), default=len(src) - 1) + 1  # out-of-line retry loops follow
bounds.append((tail, "retry loops"))
by = defaultdict(lambda: defaultdict(float))
why = defaultdict(lambda: defaultdict(float))
for k, x in enumerate(src):
    role = [n for s, n in bounds if k >= s][-1]
    if k in label:
        what = "wait " + label[k]
    elif "EXIT" in x:  # stall_barrier of warps waiting at the final __syncthreads: idle warps, finished roles
        role, what = "done", "final sync"
    else:
        what = "named barrier" if re.search(r"BAR\.(SYNC|ARV|RED)", x) else "work"
        for h in reasons:
            why[role][h] += float(data[k][ix[h]] or 0)
    by[role][what] += smp[k]
tot = sum(sum(d.values()) for d in by.values())
for role, d in sorted(by.items(), key=lambda kv: -sum(kv[1].values())):
    rt = sum(d.values())
    if rt:
        print(f"{role}: {100 * rt / tot:.1f}% of all samples")
        for what, v in sorted(d.items(), key=lambda kv: -kv[1]):
            print(f"  {100 * v / rt:5.1f}%  {what}")
            if what == "work":
                top = sorted(why[role].items(), key=lambda kv: -kv[1])[:4]
                print("         " + ", ".join(f"{h[6:]} {100 * c / rt:.1f}%" for h, c in top))
