import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps.ref import list_refs  # noqa: E402

for r in list_refs():
    track = r.get("track") or {}
    fell = track.get("fell_at_s")
    status = "" if not track else ("E_ref 不摔" if fell is None else f"E_ref 摔 @ {fell:.2f} s")
    print(f"{r['source']:12s} {r['name']:48s} {r['duration_s']:5.1f} s  {status}")
