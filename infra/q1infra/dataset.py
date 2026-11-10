"""动作库构建：一批人体动作 → 重定向 → 可行性检查 → 截掉不可行时段 → 合并为一个 any4hdmi 数据集。

用法（在 infra/ 下）：
    bin/q1py -m q1infra.dataset --name loco_v1 --bvh 'data/lafan1/walk*.bvh' \
        --smplx 'data/amass/ACCAD/*Walking*/*.npz' --exclude 'stagei|Cal' -j 8

每段动作是一个任务目录 dataset_jobs/<任务>/（与 jobs/ 同层级，manifest 里的相对路径才对得上；
已有 retarget/、check/ 的直接复用），合并结果写到
datasets/<名字>/：manifest.json、motions/<任务>__<k>.npz（只含 qpos）、index.json（每段的来源、
起止时间与可行性统计）。

截取规则：
  * 不允许的接触（非脚着地）所在帧前后各扩 0.5 s 截掉；
  * 平衡（接触力解释不了根力）连续不满足 ≥ 1.5 s 的时段截掉（更短的多是逆动力学噪声）；
  * 双脚都离地 > 4 cm 持续 ≥ 0.4 s 的时段前后各扩 0.3 s 截掉（台阶、平台、大跳；CMU 等文件名不含动作名，
    只能靠这条和上两条拦住楼梯、坐椅子这类平地上做不了的动作）；
  * 片段起点挪到第一个安稳帧（骨盆倾角 < 30°、离地最近的脚 < 2 cm、关节角速度 < 20 rad/s），前 2 s 内没有则丢弃；
  * 截完后短于 2 s 的片段丢弃；
  * 力矩通过率 < 80% 的整段丢弃（超出 Q1 能力，如冲刺、大跳）。

高动态子集（--high-dynamic，只给 E_ref）：离地一条放宽到持续 ≥ 1.0 s 才截（跳跃腾空留下，台阶、平台仍拦住）；
力矩不再整段丢弃，只截掉超过峰值 1.25 倍且持续 ≥ 0.2 s 的时段（前后各扩 0.2 s），超限帧占比 > 40% 才整段丢弃。
--novel-vs 给出已有动作库的 index.json：已在其中的源动作只输出新增帧前 2 s、后 1 s 的窗口，其余源动作整段输出。

    bin/q1py_server -m q1infra.dataset --name hd_v1 --high-dynamic --bvh 'data/lafan1/*.bvh' --existing 'amass_*' \
        --novel-vs datasets/amass_v1/index.json datasets/base_v2/index.json --exclude 'ground|fallAndGetUp|pushAndFall' -j 32
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

INFRA = Path(__file__).resolve().parents[1]
JOBS = INFRA / "dataset_jobs"
DT = 0.02
CONTACT_PAD_S = 0.5
UNSUPPORTED_MIN_S = 1.5
AIRBORNE_GAP_M = 0.04  # 双脚都离地超过此高度……
AIRBORNE_MIN_S = 0.4  # ……持续这么久（台阶、站上平台、大跳；跑步腾空约 0.1–0.2 s）
AIRBORNE_PAD_S = 0.3
MIN_PIECE_S = 2.0
# 片段起点要"安稳"：训练和整条评测都会把机器人直接放到起点姿态上，起点在剧烈动作中间（投掷发力、腾空、
# 大幅前倾）时几步内就倒。起点往后挪到第一个满足条件的帧，前 START_SEARCH_S 内找不到则丢弃该片段。
START_MAX_TILT_DEG = 30.0
START_MAX_SOLE_GAP_M = 0.02
START_MAX_JOINT_VEL = 20.0
START_SEARCH_S = 2.0
MIN_TORQUE_PASS = 0.80
HD_AIRBORNE_MIN_S = 1.0
HD_TORQUE_SCALE = 1.25  # 只截超过峰值这么多倍的时段：跟踪策略不必逐帧复现参考力矩
HD_TORQUE_MIN_S = 0.2
HD_TORQUE_PAD_S = 0.2
HD_MIN_TORQUE_PASS = 0.60
HD_LEAD_S = 2.0  # 新增帧之前带上的引导段（起点仍要安稳）
HD_TAIL_S = 1.0
HD_NEW_MIN_S = 1.0  # 新增帧连续不足这么长不算新片段
_TAU_MAX = None


def _clean(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def job_name(src: Path, kind: str) -> str:
    if kind == "bvh":
        return f"lafan_{src.stem}"
    stem = _clean(re.sub(r"_(stageii|poses)$", "", src.stem))
    if "amass_raw" in src.parts:
        # infra/data/amass_raw/<子集>/<子目录…>/<名字>_poses.npz（AMASS SMPL+H）
        i = src.parts.index("amass_raw")
        return "amass_" + "_".join(_clean(p) for p in src.parts[i + 1 : -1]) + f"_{stem}"
    folder = re.sub(r"_c3d$", "", src.parent.name)
    return f"accad_{folder}_{stem}"


def prepare(src: str, kind: str, smplx_fps: float | None) -> tuple[str, str | None]:
    """重定向 + 检查（已有的跳过）。返回 (任务名, 错误信息)。"""
    src_p = Path(src).resolve()
    job = job_name(Path(os.path.abspath(src)), kind)  # 名字按给定路径（数据可能经符号链接放在别的盘）
    job_dir = JOBS / job
    try:
        if not (job_dir / "retarget" / "manifest.json").exists():
            from q1infra.retarget.gmr_q1 import retarget_bvh, retarget_smplx

            if (job_dir / "retarget").exists():
                import shutil

                shutil.rmtree(job_dir)
            if kind == "bvh":
                retarget_bvh(src_p, job_dir, link_input=True)
            else:
                retarget_smplx(src_p, job_dir, fps=smplx_fps, link_input=True)
        if not (job_dir / "check" / "annotations.npz").exists():
            from q1infra.feasibility.check import check_path
            from q1infra.feasibility.scene import FLAT

            check_path(job_dir / "retarget", FLAT)
    except Exception as e:  # noqa: BLE001  一段失败不影响其他段
        return job, f"{type(e).__name__}: {e}"
    return job, None


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 的连续区间 [a, b)。"""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def tau_max() -> np.ndarray:
    global _TAU_MAX
    if _TAU_MAX is None:
        from q1infra.feasibility.check import load_robot
        from q1infra.feasibility.scene import FLAT

        _TAU_MAX = load_robot(FLAT).tau_max
    return _TAU_MAX


def over_torque(ann: dict) -> np.ndarray:
    """(T,) 高动态规则下力矩超限的帧（任一关节超过峰值 HD_TORQUE_SCALE 倍）。"""
    return (np.abs(ann["joint_torque"]) > HD_TORQUE_SCALE * tau_max()).any(axis=1)


def keep_mask(ann: dict, n: int, high_dynamic: bool = False) -> np.ndarray:
    bad = np.zeros(n, bool)
    pad = int(round(CONTACT_PAD_S / DT))
    for a, b in _runs(ann["disallowed_contact"]):
        bad[max(0, a - pad) : min(n, b + pad)] = True
    for a, b in _runs(ann["unsupported"]):
        if (b - a) * DT >= UNSUPPORTED_MIN_S:
            bad[a:b] = True
    pad = int(round(AIRBORNE_PAD_S / DT))
    air_min = HD_AIRBORNE_MIN_S if high_dynamic else AIRBORNE_MIN_S
    for a, b in _runs(ann["foot_gap_m"].min(axis=1) > AIRBORNE_GAP_M):
        if (b - a) * DT >= air_min:
            bad[max(0, a - pad) : min(n, b + pad)] = True
    if high_dynamic:
        pad = int(round(HD_TORQUE_PAD_S / DT))
        for a, b in _runs(over_torque(ann)):
            if (b - a) * DT >= HD_TORQUE_MIN_S:
                bad[max(0, a - pad) : min(n, b + pad)] = True
    return ~bad


def calm_frames(qpos: np.ndarray, foot_gap: np.ndarray) -> np.ndarray:
    """(T,) 可作片段起点的帧：骨盆倾角、离地最近的脚底高度、关节角速度都在 START_* 以内。"""
    x, y = qpos[:, 4], qpos[:, 5]
    up_z = 1.0 - 2.0 * (x * x + y * y)  # 骨盆 z 轴在世界系的竖直分量
    tilt = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
    jv = np.abs(np.gradient(qpos[:, 7:], DT, axis=0)).max(axis=1)
    return (tilt < START_MAX_TILT_DEG) & (foot_gap.min(axis=1) < START_MAX_SOLE_GAP_M) & (jv < START_MAX_JOINT_VEL)


def novel_windows(keep: np.ndarray, covered: np.ndarray) -> list[tuple[int, int]]:
    """keep 里不在 covered 中的连续帧（≥ HD_NEW_MIN_S），前后扩 HD_LEAD_S / HD_TAIL_S 后截在所属 keep 段内，重叠的合并。"""
    lead, tail = int(round(HD_LEAD_S / DT)), int(round(HD_TAIL_S / DT))
    out: list[tuple[int, int]] = []
    for ka, kb in _runs(keep):
        for a, b in _runs(keep[ka:kb] & ~covered[ka:kb]):
            if (b - a) * DT < HD_NEW_MIN_S:
                continue
            a, b = max(ka, ka + a - lead), min(kb, ka + b + tail)
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
    return out


def collect(job: str, high_dynamic: bool = False, covered_s: list[tuple[float, float]] | None = None) -> tuple[list[dict], dict]:
    """covered_s：该源动作已在其他动作库中的时段（秒）；给出时只输出新增帧附近的窗口（高动态模式）。"""
    from any4hdmi.core.format import load_manifest, load_motion

    job_dir = JOBS / job
    man = load_manifest(job_dir / "retarget")
    (motion,) = sorted((man.root / "motions").rglob("*.npz"))
    qpos = load_motion(motion)
    report = json.loads((job_dir / "check" / "report.json").read_text())
    keys = ("disallowed_contact", "unsupported", "foot_gap_m") + (("joint_torque",) if high_dynamic else ())
    with np.load(job_dir / "check" / "annotations.npz") as d:
        ann = {k: d[k] for k in keys}
    info = {
        "job": job, "duration_s": len(qpos) * DT, "verdict": report["verdict"],
        "torque_pass": report["torque"]["pass_rate"], "balance_pass": report["balance"]["pass_rate"],
    }
    if high_dynamic:
        info["torque_pass_hd"] = round(float(1 - over_torque(ann).mean()), 4)
        if info["torque_pass_hd"] < HD_MIN_TORQUE_PASS:
            info["dropped"] = "torque"
            return [], info
    elif report["torque"]["pass_rate"] < MIN_TORQUE_PASS:
        info["dropped"] = "torque"
        return [], info
    n = len(qpos)
    keep = keep_mask(ann, n, high_dynamic)
    if covered_s is not None:
        covered = np.zeros(n, bool)
        for s, e in covered_s:
            covered[int(round(s / DT)) : int(round(e / DT))] = True
        spans = novel_windows(keep, covered)
    else:
        spans = _runs(keep)
    calm = calm_frames(qpos, ann["foot_gap_m"])
    pieces = []
    for a, b in spans:
        ok = np.flatnonzero(calm[a : min(b, a + int(START_SEARCH_S / DT))])
        if ok.size == 0:
            continue
        a += int(ok[0])
        if (b - a) * DT >= MIN_PIECE_S:
            pieces.append({"job": job, "start_s": round(a * DT, 2), "end_s": round(b * DT, 2), "qpos": qpos[a:b]})
    info["kept_s"] = round(sum(p["end_s"] - p["start_s"] for p in pieces), 2)
    return pieces, info


def main() -> None:
    p = argparse.ArgumentParser(description="批量重定向 + 可行性检查 + 截取，合并为一个 any4hdmi 动作库")
    p.add_argument("--name", required=True, help="输出 datasets/<名字>/")
    p.add_argument("--bvh", nargs="*", default=[], help="LAFAN1 BVH（glob）")
    p.add_argument("--smplx", nargs="*", default=[], help="AMASS SMPL-X npz（glob）")
    p.add_argument("--exclude", default=None, help="文件路径匹配此正则的跳过")
    p.add_argument("--smplx-fps", type=float, default=None)
    p.add_argument("--existing", nargs="*", default=[], help="直接复用 dataset_jobs/ 下已做完重定向和检查的任务（任务名 glob），源文件可以已删除")
    p.add_argument("--high-dynamic", action="store_true", help="高动态子集的放宽规则")
    p.add_argument("--novel-vs", nargs="*", default=[], help="已有动作库的 index.json：其中的源动作只输出新增帧附近的窗口")
    p.add_argument("-j", "--jobs", type=int, default=8, help="并行进程数")
    args = p.parse_args()

    srcs: list[tuple[str, str]] = []
    for kind, pats in (("bvh", args.bvh), ("smplx", args.smplx)):
        for pat in pats:
            for f in sorted(glob.glob(pat)):
                if args.exclude and re.search(args.exclude, f):
                    continue
                srcs.append((f, kind))
    print(f"{len(srcs)} 段源动作", flush=True)

    jobs, errors = [], {}
    pending = list(srcs)
    done_n = 0
    for attempt in range(5):  # 某个工作进程崩溃（如 MuJoCo 段错误）会打断整个进程池：重建进程池，接着做剩下的
        if not pending:
            break
        with ProcessPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(prepare, f, k, args.smplx_fps): (f, k) for f, k in pending}
            try:
                for fut in as_completed(futs):
                    job, err = fut.result()
                    pending.remove(futs[fut])
                    done_n += 1
                    if err:
                        errors[job] = err
                        print(f"[{done_n}/{len(srcs)}] {job} 失败：{err}", flush=True)
                    else:
                        jobs.append(job)
                        print(f"[{done_n}/{len(srcs)}] {job}", flush=True)
            except BrokenProcessPool:
                print(f"进程池中断（第 {attempt + 1} 次），剩余 {len(pending)} 段重新提交", flush=True)
    for f, k in pending:
        errors[job_name(Path(os.path.abspath(f)), k)] = "工作进程崩溃"
    for pat in args.existing:
        for d in sorted(JOBS.glob(pat)):
            if args.exclude and re.search(args.exclude, d.name):
                continue
            if (d / "check" / "annotations.npz").exists() and d.name not in jobs:
                jobs.append(d.name)
    covered: dict[str, list[tuple[float, float]]] = {}
    for f in args.novel_vs:
        for pc in json.loads(Path(f).read_text())["pieces"]:
            covered.setdefault(pc["job"], []).append((pc["start_s"], pc["end_s"]))

    from any4hdmi.core.format import save_motion, write_manifest

    out = INFRA / "datasets" / args.name
    if out.exists():
        import shutil

        shutil.rmtree(out)
    (out / "motions").mkdir(parents=True)
    index, infos, frames = [], [], 0
    qpos_names = None
    for job in sorted(jobs):
        pieces, info = collect(job, args.high_dynamic, covered.get(job) if args.novel_vs else None)
        infos.append(info)
        if qpos_names is None:
            qpos_names = json.loads((JOBS / job / "retarget" / "manifest.json").read_text())["qpos_names"]
        for k, pc in enumerate(pieces):
            name = f"{job}__{k}"
            save_motion(out / "motions" / f"{name}.npz", pc.pop("qpos").astype(np.float32))
            frames += int(round((pc["end_s"] - pc["start_s"]) / DT))
            index.append({"name": name, **pc})
    from q1infra.retarget.gmr_q1 import Q1_MJCF

    rules = {
        "contact_pad_s": CONTACT_PAD_S, "unsupported_min_s": UNSUPPORTED_MIN_S, "airborne": [AIRBORNE_GAP_M, AIRBORNE_MIN_S, AIRBORNE_PAD_S],
        "min_piece_s": MIN_PIECE_S, "min_torque_pass": MIN_TORQUE_PASS}
    if args.high_dynamic:
        rules.update({"airborne": [AIRBORNE_GAP_M, HD_AIRBORNE_MIN_S, AIRBORNE_PAD_S], "min_torque_pass": None,
                      "torque_cut": [HD_TORQUE_SCALE, HD_TORQUE_MIN_S, HD_TORQUE_PAD_S], "min_torque_pass_hd": HD_MIN_TORQUE_PASS})
    if args.novel_vs:
        rules["novel_vs"] = {"indices": args.novel_vs, "lead_s": HD_LEAD_S, "tail_s": HD_TAIL_S, "new_min_s": HD_NEW_MIN_S}
    write_manifest(
        out, dataset_name=args.name, mjcf=Q1_MJCF, timestep=DT, qpos_names=qpos_names, num_motions=len(index),
        source={"builder": "q1infra.dataset", "sources": len(jobs), "rules": rules},
        total_hours=frames * DT / 3600,
    )
    (out / "index.json").write_text(json.dumps({"pieces": index, "jobs": infos, "errors": errors}, indent=1, ensure_ascii=False) + "\n")
    src_h = sum(i["duration_s"] for i in infos) / 3600
    dropped = [i["job"] for i in infos if i.get("dropped")]
    print(f"\n{out}: {len(index)} 段，{frames * DT / 3600:.2f} h（源 {src_h:.2f} h）；整段丢弃 {len(dropped)}：{dropped}；失败 {len(errors)}")


if __name__ == "__main__":
    main()
