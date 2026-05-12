#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import site
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

_npp_lib = Path(site.getsitepackages()[0]) / "nvidia" / "npp" / "lib"
_npp_so = _npp_lib / "libnppicc.so.12"
if _npp_so.is_file():
    ctypes.CDLL(str(_npp_so), mode=ctypes.RTLD_GLOBAL)

import torch
from tqdm import tqdm

_openai_client = None

GPT_JUDGE_SYSTEM = """\
You are a structured-output extractor. The user will give you a model's free-text \
response about audio-video synchronization. Extract the following fields and return \
ONLY valid JSON (no markdown, no explanation):

{"synced": <bool>, "direction": "none"|"delay"|"early", "offset_sec": <float>, "t_v": <float or null>, "t_a": <float or null>, "explanation": "<one sentence>"}

Rules:
- synced: true if the model says audio and video are synchronized, false otherwise.
- direction: "delay" means audio comes AFTER the visual event; "early" means audio \
comes BEFORE the visual event; "none" if synced is true.
- offset_sec: estimated time gap in seconds. 0.0 if synced.
- t_v: the timestamp (in seconds) the model attributes to the VISUAL event. null if not mentioned.
- t_a: the timestamp (in seconds) the model attributes to the AUDIO event. null if not mentioned.
- If you cannot determine a field, use the default (true / "none" / 0.0 / null / "").
"""


def _get_openai_client(api_key: Optional[str] = None):
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    _openai_client = OpenAI(api_key=key)
    return _openai_client


def gpt_extract_prediction(
    raw_output: str,
    api_key: Optional[str] = None,
    model: str = "gpt-5.4",
) -> Optional[Dict[str, Any]]:
    client = _get_openai_client(api_key)
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GPT_JUDGE_SYSTEM},
                {"role": "user", "content": raw_output},
            ],
            temperature=0.0,
            max_completion_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        for pat in [
            re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL),
            re.compile(r"(\{.*?\})", re.DOTALL),
        ]:
            m = pat.search(text)
            if m:
                obj = json.loads(m.group(1))
                synced = obj.get("synced")
                if isinstance(synced, str):
                    synced = synced.lower() in ("true", "yes", "1")
                direction = str(obj.get("direction", "none")).lower().strip()
                if direction not in ("delay", "early", "none"):
                    direction = "none"
                t_v_raw = obj.get("t_v")
                t_a_raw = obj.get("t_a")
                pred_t_v = float(t_v_raw) if t_v_raw is not None else None
                pred_t_a = float(t_a_raw) if t_a_raw is not None else None
                return {
                    "pred_synced": bool(synced),
                    "pred_direction": direction,
                    "pred_offset_sec": float(obj.get("offset_sec", 0.0)),
                    "pred_t_v": pred_t_v,
                    "pred_t_a": pred_t_a,
                    "pred_explanation": str(obj.get("explanation", "")),
                    "parse_method": "gpt_judge",
                }
    except Exception as exc:
        print(f"  [gpt-judge] API error: {exc}", flush=True)
    return None

DATA_ROOT = Path("./data/video_source")
ORIGINAL_ROOT = DATA_ROOT / "original"
AUDIO_ROOT = DATA_ROOT / "extracted_audio" / "original"


def set_data_root(root: Path) -> None:
    global DATA_ROOT, ORIGINAL_ROOT, AUDIO_ROOT
    DATA_ROOT = root.resolve()
    ORIGINAL_ROOT = DATA_ROOT / "original"
    AUDIO_ROOT = DATA_ROOT / "extracted_audio" / "original"

EVAL_PROMPT = """\
Watch this video and listen to its audio carefully. \
Determine whether the audio and video tracks are synchronized. \
If they are not synchronized, identify the direction of the offset \
(audio delayed or audio early relative to video) and estimate the offset in seconds. \
Explain your reasoning."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate sync model on test set.")
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--adapter", type=str, default=None)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data/video_source"),
    )
    p.add_argument(
        "--test-jsonl",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--gpt-judge", action="store_true", default=False)
    p.add_argument("--openai-api-key", type=str, default=None)
    p.add_argument("--gpt-model", type=str, default="gpt-5.4")
    p.add_argument("--vllm", action="store_true", default=False)
    p.add_argument("--tp", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=65536)
    return p.parse_args()


def parse_ground_truth(video_field: str) -> Dict[str, Any]:
    m_delay = re.search(r"_delay_([\d.]+)s\.mp4", video_field)
    m_early = re.search(r"_early_([\d.]+)s\.mp4", video_field)
    if m_delay:
        return {"synced": False, "direction": "delay", "offset_sec": float(m_delay.group(1))}
    elif m_early:
        return {"synced": False, "direction": "early", "offset_sec": float(m_early.group(1))}
    else:
        return {"synced": True, "direction": "none", "offset_sec": 0.0}


def resolve_video_path(video_field: str) -> str:
    if os.path.isabs(video_field) and os.path.exists(video_field):
        return video_field
    candidate_dirs = [
        ORIGINAL_ROOT / "uag_oops",
        DATA_ROOT / "random_shift_video" / "delay",
        DATA_ROOT / "random_shift_video" / "early",
        ORIGINAL_ROOT,
    ]
    for d in candidate_dirs:
        c = d / video_field
        if c.exists():
            return str(c)
    return str(ORIGINAL_ROOT / "uag_oops" / video_field)


def resolve_audio_path(video_path: str) -> str:
    video_p = Path(video_path)
    try:
        rel = video_p.relative_to(DATA_ROOT)
    except ValueError:
        rel = Path(video_p.name)
    audio_path = DATA_ROOT / "extracted_audio" / rel.with_suffix(".wav")
    if audio_path.exists():
        return str(audio_path)
    base_stem = re.sub(r"_(delay|early)_[\d.]+s$", "", video_p.stem)
    fallback = DATA_ROOT / "extracted_audio" / "original" / "uag_oops" / (base_stem + ".wav")
    if fallback.exists():
        return str(fallback)
    return str(audio_path)


def extract_timestamps(text: str) -> Tuple[Optional[float], Optional[float]]:
    text_lower = text.lower()
    all_times = [(m.start(), float(m.group(1)))
                 for m in re.finditer(r"(?:at|around|about)\s+([\d]+\.?\d*)\s*s", text_lower)]
    if len(all_times) >= 2:
        return (all_times[0][1], all_times[1][1])
    if len(all_times) == 1:
        return (all_times[0][1], all_times[0][1])
    return (None, None)


def load_test_data(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            video_path = resolve_video_path(obj["video"])
            audio_path = resolve_audio_path(video_path)
            gt = parse_ground_truth(obj["video"])
            gt_t_v, gt_t_a = extract_timestamps(obj.get("chosen", ""))
            data.append({
                "video": obj["video"],
                "video_path": video_path,
                "audio_path": audio_path,
                "prompt": obj["prompt"],
                "chosen": obj["chosen"],
                "rejected": obj["rejected"],
                "gt_synced": gt["synced"],
                "gt_direction": gt["direction"],
                "gt_offset_sec": gt["offset_sec"],
                "gt_t_v": gt_t_v,
                "gt_t_a": gt_t_a,
            })
    if max_samples > 0:
        data = data[:max_samples]
    return data


def extract_prediction(text: str) -> Dict[str, Any]:
    text = text.strip()

    for pattern in [
        re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL),
        re.compile(r"(\{[^{}]*\"synced\"[^{}]*\})", re.DOTALL),
        re.compile(r"(\{.*?\})", re.DOTALL),
    ]:
        m = pattern.search(text)
        if m:
            try:
                obj = json.loads(m.group(1))
                synced = obj.get("synced")
                if isinstance(synced, str):
                    synced = synced.lower() in ("true", "yes", "1")
                direction = str(obj.get("direction", "none")).lower().strip()
                if direction not in ("delay", "early", "none"):
                    direction = "none"
                offset = float(obj.get("offset_sec", 0.0))
                explanation = str(obj.get("explanation", ""))
                t_v_raw = obj.get("t_v")
                t_a_raw = obj.get("t_a")
                return {
                    "pred_synced": bool(synced),
                    "pred_direction": direction,
                    "pred_offset_sec": offset,
                    "pred_t_v": float(t_v_raw) if t_v_raw is not None else None,
                    "pred_t_a": float(t_a_raw) if t_a_raw is not None else None,
                    "pred_explanation": explanation,
                    "parse_method": "json",
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    text_lower = text.lower()
    synced = None
    direction = "none"
    offset = 0.0

    pred_t_v, pred_t_a = extract_timestamps(text)

    desync_kws = [
        "not synchronized", "not aligned", "desync", "mismatch", "misalign",
        "not in sync", "out of sync", "clearly not", "not well aligned",
        "are not aligned", "audio and visual event are clearly not",
    ]
    sync_kws = [
        "synchronized", "well aligned", "well-aligned", "in sync",
        "appear synchronized", "appears synchronized", "closely aligned",
        "audio and video are aligned", "matches closely",
    ]
    if any(kw in text_lower for kw in desync_kws):
        synced = False
    elif any(kw in text_lower for kw in sync_kws):
        synced = True

    if synced is False:
        delay_kws = ["audio delayed", "audio lags", "audio comes after", "sound comes after",
                     "sound is heard later", "audio is delayed", "sound follows"]
        early_kws = ["audio early", "audio leads", "audio comes before", "sound comes before",
                     "audio precedes", "sound is heard before", "sound precedes", "audio is early"]
        if any(kw in text_lower for kw in delay_kws):
            direction = "delay"
        elif any(kw in text_lower for kw in early_kws):
            direction = "early"

        if direction == "none" and pred_t_v is not None and pred_t_a is not None and pred_t_v != pred_t_a:
            if pred_t_a > pred_t_v:
                direction = "delay"
            else:
                direction = "early"
            offset = abs(pred_t_a - pred_t_v)

        if offset == 0.0:
            offset_match = re.search(
                r"(?:gap|separation|offset|mismatch|differ\w*)\s*(?:of\s+)?(?:about\s+|roughly\s+|approximately\s+)?"
                r"([\d]+\.?\d*)\s*s",
                text_lower,
            )
            if not offset_match:
                offset_match = re.search(
                    r"(?:about\s+|roughly\s+|approximately\s+)?([\d]+\.?\d*)\s*s\s*"
                    r"(?:gap|separation|offset|mismatch|differ)",
                    text_lower,
                )
            if offset_match:
                offset = float(offset_match.group(1))

    if synced is None:
        synced = True

    return {
        "pred_synced": synced,
        "pred_direction": direction,
        "pred_offset_sec": offset,
        "pred_t_v": pred_t_v,
        "pred_t_a": pred_t_a,
        "pred_explanation": "",
        "parse_method": "regex_fallback",
    }


def load_model(base_model: str, adapter: Optional[str]):
    from multi_omni_adapter import get_adapter

    omni = get_adapter(base_model, adapter)
    omni.load()
    return omni


def run_inference(omni, video_path: str, audio_path: str,
                  max_new_tokens: int, temperature: float) -> str:
    return omni.infer(video_path, audio_path, EVAL_PROMPT, max_new_tokens, temperature)


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def preprocess_video_for_vllm(video_path: str):
    from qwen_omni_utils import process_mm_info
    import numpy as np

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": 2.0, "max_frames": 128},
            {"type": "text", "text": "placeholder"},
        ],
    }]
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    video_tensor = videos[0]
    return (video_tensor * 255).byte().numpy()


def preprocess_audio_for_vllm(audio_path: str, target_sr: int = 16000):
    import numpy as np
    import wave

    with wave.open(audio_path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != target_sr:
        duration = len(x) / sr
        new_len = int(duration * target_sr)
        x = np.interp(
            np.linspace(0, len(x) - 1, new_len),
            np.arange(len(x)),
            x,
        )
    return x, target_sr


def build_vllm_prompt(question: str, base_model: str) -> str:
    from omni_model_loading import vllm_user_mm_prefix

    mm = vllm_user_mm_prefix(base_model, include_audio=True)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{mm}"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    sync_correct = sum(1 for r in results if r["pred_synced"] == r["gt_synced"])
    sync_acc = sync_correct / total

    desync_samples = [r for r in results if not r["gt_synced"]]
    if desync_samples:
        dir_correct = sum(1 for r in desync_samples if r["pred_direction"] == r["gt_direction"])
        dir_acc = dir_correct / len(desync_samples)
    else:
        dir_acc = None

    def label(r, prefix):
        if r[f"{prefix}synced"]:
            return "synced"
        return r[f"{prefix}direction"]
    three_class_correct = sum(1 for r in results if label(r, "pred_") == label(r, "gt_"))
    three_class_acc = three_class_correct / total

    offset_errors = []
    for r in desync_samples:
        if not r["pred_synced"] and r["pred_offset_sec"] > 0:
            offset_errors.append(abs(r["pred_offset_sec"] - r["gt_offset_sec"]))
    offset_mae = mean(offset_errors) if offset_errors else None
    offset_median = median(offset_errors) if offset_errors else None

    synced_samples = [r for r in results if r["gt_synced"]]
    delay_samples = [r for r in results if r["gt_direction"] == "delay"]
    early_samples = [r for r in results if r["gt_direction"] == "early"]

    synced_acc = (sum(1 for r in synced_samples if r["pred_synced"]) / len(synced_samples)) if synced_samples else None
    delay_acc = (sum(1 for r in delay_samples if not r["pred_synced"] and r["pred_direction"] == "delay") / len(delay_samples)) if delay_samples else None
    early_acc = (sum(1 for r in early_samples if not r["pred_synced"] and r["pred_direction"] == "early") / len(early_samples)) if early_samples else None

    within_05 = sum(1 for e in offset_errors if e <= 0.5) if offset_errors else 0
    within_10 = sum(1 for e in offset_errors if e <= 1.0) if offset_errors else 0

    json_parsed = sum(1 for r in results if r.get("parse_method") == "json")
    regex_parsed = sum(1 for r in results if r.get("parse_method") == "regex_fallback")
    gpt_parsed = sum(1 for r in results if r.get("parse_method") == "gpt_judge")

    tv_errors = []
    ta_errors = []
    for r in results:
        gt_tv = r.get("gt_t_v")
        gt_ta = r.get("gt_t_a")
        pred_tv = r.get("pred_t_v")
        pred_ta = r.get("pred_t_a")
        if gt_tv is not None and pred_tv is not None:
            tv_errors.append(abs(pred_tv - gt_tv))
        if gt_ta is not None and pred_ta is not None:
            ta_errors.append(abs(pred_ta - gt_ta))
    tv_mae = round(mean(tv_errors), 4) if tv_errors else None
    ta_mae = round(mean(ta_errors), 4) if ta_errors else None
    tv_median = round(median(tv_errors), 4) if tv_errors else None
    ta_median = round(median(ta_errors), 4) if ta_errors else None

    return {
        "total_samples": total,
        "sync_desync_accuracy": round(sync_acc, 4),
        "three_class_accuracy": round(three_class_acc, 4),
        "direction_accuracy_on_desync": round(dir_acc, 4) if dir_acc is not None else None,
        "per_category": {
            "synced_accuracy": round(synced_acc, 4) if synced_acc is not None else None,
            "delay_accuracy": round(delay_acc, 4) if delay_acc is not None else None,
            "early_accuracy": round(early_acc, 4) if early_acc is not None else None,
            "synced_count": len(synced_samples),
            "delay_count": len(delay_samples),
            "early_count": len(early_samples),
        },
        "offset_mae_sec": round(offset_mae, 4) if offset_mae is not None else None,
        "offset_median_sec": round(offset_median, 4) if offset_median is not None else None,
        "offset_within_0.5s": within_05,
        "offset_within_1.0s": within_10,
        "offset_evaluated_count": len(offset_errors),
        "timestamp_tv_mae_sec": tv_mae,
        "timestamp_ta_mae_sec": ta_mae,
        "timestamp_tv_median_sec": tv_median,
        "timestamp_ta_median_sec": ta_median,
        "timestamp_evaluated_tv": len(tv_errors),
        "timestamp_evaluated_ta": len(ta_errors),
        "parse_stats": {"json": json_parsed, "regex_fallback": regex_parsed, "gpt_judge": gpt_parsed},
    }


def print_summary(metrics: Dict[str, Any], label: str) -> None:
    print()
    print(f"{'=' * 60}")
    print(f"  Eval Summary: {label}")
    print(f"{'=' * 60}")
    print(f"  Total samples:            {metrics['total_samples']}")
    print(f"  Sync/Desync Accuracy:     {metrics['sync_desync_accuracy']:.1%}")
    print(f"  3-Class Accuracy:         {metrics['three_class_accuracy']:.1%}")
    if metrics["direction_accuracy_on_desync"] is not None:
        print(f"  Direction Acc (desync):    {metrics['direction_accuracy_on_desync']:.1%}")
    print(f"  ─── Per Category ───")
    pc = metrics["per_category"]
    if pc["synced_accuracy"] is not None:
        print(f"    Synced correct:         {pc['synced_accuracy']:.1%}  ({pc['synced_count']} samples)")
    if pc["delay_accuracy"] is not None:
        print(f"    Delay correct:          {pc['delay_accuracy']:.1%}  ({pc['delay_count']} samples)")
    if pc["early_accuracy"] is not None:
        print(f"    Early correct:          {pc['early_accuracy']:.1%}  ({pc['early_count']} samples)")
    print(f"  ─── Offset Estimation ───")
    if metrics["offset_mae_sec"] is not None:
        print(f"    MAE:                    {metrics['offset_mae_sec']:.3f}s")
        print(f"    Median Error:           {metrics['offset_median_sec']:.3f}s")
        print(f"    Within 0.5s:            {metrics['offset_within_0.5s']} / {metrics['offset_evaluated_count']}")
        print(f"    Within 1.0s:            {metrics['offset_within_1.0s']} / {metrics['offset_evaluated_count']}")
    else:
        print(f"    (no valid offset predictions)")
    print(f"  ─── Timestamp Estimation ───")
    if metrics.get("timestamp_tv_mae_sec") is not None:
        print(f"    t_v MAE:                {metrics['timestamp_tv_mae_sec']:.3f}s  ({metrics['timestamp_evaluated_tv']} samples)")
        print(f"    t_v Median Error:       {metrics['timestamp_tv_median_sec']:.3f}s")
    else:
        print(f"    t_v: (no valid pairs)")
    if metrics.get("timestamp_ta_mae_sec") is not None:
        print(f"    t_a MAE:                {metrics['timestamp_ta_mae_sec']:.3f}s  ({metrics['timestamp_evaluated_ta']} samples)")
        print(f"    t_a Median Error:       {metrics['timestamp_ta_median_sec']:.3f}s")
    else:
        print(f"    t_a: (no valid pairs)")
    print(f"  ─── Parse Stats ───")
    ps = metrics["parse_stats"]
    print(f"    JSON parsed:            {ps['json']}")
    print(f"    GPT judge:              {ps.get('gpt_judge', 0)}")
    print(f"    Regex fallback:         {ps['regex_fallback']}")
    print(f"{'=' * 60}")


def main() -> None:
    args = parse_args()
    set_data_root(args.data_root)
    test_jsonl = args.test_jsonl or (DATA_ROOT / "test.jsonl")
    output_dir = args.output_dir or Path("./eval_results/sync")

    if args.gpt_judge:
        client = _get_openai_client(args.openai_api_key)
        if client is None:
            print("[ERROR] --gpt-judge requires OPENAI_API_KEY env var or --openai-api-key argument.")
            raise SystemExit(1)
        try:
            test_resp = client.chat.completions.create(
                model=args.gpt_model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_completion_tokens=5,
            )
            print(f"[gpt-judge] API verified. Model: {args.gpt_model}")
        except Exception as exc:
            print(f"[ERROR] GPT API check failed: {exc}")
            raise SystemExit(1)

    label = args.label or (Path(args.adapter).name if args.adapter else Path(args.base_model).name)

    out_dir = output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = out_dir / "eval_results.jsonl"
    metrics_json = out_dir / "metrics.json"
    summary_txt = out_dir / "summary.txt"

    test_data = load_test_data(test_jsonl, args.max_samples)
    print(f"[data] Loaded {len(test_data)} test samples")

    processed = set()
    if results_jsonl.exists():
        with open(results_jsonl) as f:
            for line in f:
                obj = json.loads(line)
                processed.add(obj["video"])
        print(f"[resume] {len(processed)} already processed, skipping")

    def _do_extract(raw_output: str) -> Dict[str, Any]:
        if args.gpt_judge and raw_output:
            gpt_pred = gpt_extract_prediction(
                raw_output, api_key=args.openai_api_key, model=args.gpt_model,
            )
            if gpt_pred is not None:
                return gpt_pred
        return extract_prediction(raw_output)

    def _build_result(item: Dict, pred: Dict, raw_output: str) -> Dict:
        return {
            "video": item["video"],
            "video_path": item["video_path"],
            "gt_synced": item["gt_synced"],
            "gt_direction": item["gt_direction"],
            "gt_offset_sec": item["gt_offset_sec"],
            "gt_t_v": item["gt_t_v"],
            "gt_t_a": item["gt_t_a"],
            "pred_synced": pred["pred_synced"],
            "pred_direction": pred["pred_direction"],
            "pred_offset_sec": pred["pred_offset_sec"],
            "pred_t_v": pred.get("pred_t_v"),
            "pred_t_a": pred.get("pred_t_a"),
            "pred_explanation": pred.get("pred_explanation", ""),
            "parse_method": pred["parse_method"],
            "raw_output": raw_output,
        }

    use_vllm = args.vllm

    if use_vllm:
        from vllm import LLM, SamplingParams

        tp = args.tp or torch.cuda.device_count()
        todo = [item for item in test_data if item["video"] not in processed]

        print(f"[vllm] Preprocessing {len(todo)} samples (video + audio) ...")
        preprocessed_v: Dict[str, Any] = {}
        preprocessed_a: Dict[str, Any] = {}
        failed_paths: set = set()

        unique_videos = list(dict.fromkeys(item["video_path"] for item in todo))
        unique_audios = list(dict.fromkeys(item["audio_path"] for item in todo))

        for vp in tqdm(unique_videos, desc="Preprocess video", unit="video"):
            if vp in failed_paths:
                continue
            try:
                preprocessed_v[vp] = preprocess_video_for_vllm(vp)
            except Exception as e:
                failed_paths.add(vp)
                print(f"  [skip] video preprocess error: {Path(vp).name}: {e}")

        for ap in tqdm(unique_audios, desc="Preprocess audio", unit="audio"):
            if ap in failed_paths:
                continue
            try:
                preprocessed_a[ap] = preprocess_audio_for_vllm(ap)
            except Exception as e:
                failed_paths.add(ap)
                print(f"  [skip] audio preprocess error: {Path(ap).name}: {e}")

        n_skip = sum(1 for item in todo
                     if item["video_path"] in failed_paths or item["audio_path"] in failed_paths)
        if failed_paths:
            print(f"[vllm] Preprocess failed for {len(failed_paths)} path(s), "
                  f"{n_skip} sample(s) will be skipped.")

        from omni_model_loading import cap_vllm_max_model_len

        vllm_max_len = cap_vllm_max_model_len(args.base_model, args.max_model_len)
        print(f"[vllm] Loading {args.base_model} with tp={tp} (max_model_len={vllm_max_len}) ...")
        llm = LLM(
            model=args.base_model,
            tensor_parallel_size=tp,
            max_model_len=vllm_max_len,
            max_num_seqs=4,
            limit_mm_per_prompt={"video": 1, "audio": 1},
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature if args.temperature > 0 else 0.0,
            top_p=0.9 if args.temperature > 0 else 1.0,
            max_tokens=args.max_new_tokens,
        )

        vllm_todo = [item for item in todo
                     if item["video_path"] not in failed_paths
                     and item["audio_path"] not in failed_paths]
        fallback_items = [item for item in todo
                          if item["video_path"] in failed_paths
                          or item["audio_path"] in failed_paths]
        print(f"[vllm] {len(vllm_todo)} samples ready, {len(fallback_items)} deferred to transformers ...")

        for i, item in enumerate(vllm_todo):
            if item["video"] in processed:
                continue
            inp = {
                "prompt": build_vllm_prompt(EVAL_PROMPT, args.base_model),
                "multi_modal_data": {
                    "video": preprocessed_v[item["video_path"]],
                    "audio": preprocessed_a[item["audio_path"]],
                },
            }
            try:
                outputs = llm.generate([inp], sampling_params=sampling_params)
                raw_output = outputs[0].outputs[0].text.strip()
            except (ValueError, RuntimeError) as exc:
                if "longer than the maximum model length" in str(exc):
                    print(f"  [too long] {item['video']} -> fallback")
                    fallback_items.append(item)
                    continue
                else:
                    print(f"  [error] {item['video']}: {exc}")
                    raw_output = ""

            pred = _do_extract(raw_output)
            result = _build_result(item, pred, raw_output)

            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed.add(item["video"])

            if (i + 1) % 100 == 0:
                print(f"  [vllm] [{i+1}/{len(vllm_todo)}] done, {len(fallback_items)} deferred")

        preprocessed_v.clear()
        preprocessed_a.clear()

        if fallback_items:
            print(f"[fallback] Running {len(fallback_items)} samples with transformers ...")
            del llm
            gc.collect()
            torch.cuda.empty_cache()

            omni = load_model(args.base_model, args.adapter)
            for item in tqdm(fallback_items, desc="Fallback", unit="q"):
                if item["video"] in processed:
                    continue
                try:
                    raw_output = run_inference(
                        omni, item["video_path"], item["audio_path"],
                        args.max_new_tokens, args.temperature,
                    )
                except Exception as exc:
                    import traceback
                    print(f"  [error] {item['video']}: {exc}")
                    traceback.print_exc()
                    raw_output = ""

                pred = _do_extract(raw_output)
                result = _build_result(item, pred, raw_output)

                with open(results_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed.add(item["video"])
                gc.collect()
                torch.cuda.empty_cache()

    else:
        todo = [it for it in test_data if it["video"] not in processed]
        if not todo:
            print(f"[resume] all {len(test_data)} samples already done — skipping model load")
            omni = None
        else:
            omni = load_model(args.base_model, args.adapter)

        for item in tqdm(test_data, desc="Evaluating", unit="sample"):
            if item["video"] in processed:
                continue

            if not os.path.exists(item["video_path"]):
                print(f"  [skip] Video not found: {item['video_path']}")
                continue

            try:
                raw_output = run_inference(
                    omni, item["video_path"], item["audio_path"],
                    args.max_new_tokens, args.temperature,
                )
            except Exception as exc:
                import traceback
                print(f"  [error] {item['video']}: {exc}")
                traceback.print_exc()
                continue
            if not raw_output:
                print(f"  [skip] empty output for {item['video']}; will retry next run")
                continue

            pred = _do_extract(raw_output)
            result = _build_result(item, pred, raw_output)

            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            processed.add(item["video"])
            gc.collect()
            torch.cuda.empty_cache()

    all_results = []
    if results_jsonl.exists():
        with open(results_jsonl) as f:
            for line in f:
                all_results.append(json.loads(line))

    metrics = compute_metrics(all_results)
    metrics["eval_config"] = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "data_root": str(args.data_root),
        "test_jsonl": str(test_jsonl),
        "total_test_samples": len(test_data),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "gpt_judge": args.gpt_judge,
        "gpt_model": args.gpt_model if args.gpt_judge else None,
        "vllm": args.vllm,
    }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print_summary(metrics, label)

    with open(summary_txt, "w", encoding="utf-8") as f:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_summary(metrics, label)
        f.write(buf.getvalue())

    print(f"\n[output] Results JSONL: {results_jsonl}")
    print(f"[output] Metrics JSON:  {metrics_json}")
    print(f"[output] Summary:       {summary_txt}")


if __name__ == "__main__":
    main()
