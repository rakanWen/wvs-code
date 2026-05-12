#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re

if not (os.environ.get("FORCE_QWENVL_VIDEO_READER") or "").strip():
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
if not (os.environ.get("AV_LOG_LEVEL") or "").strip():
    os.environ["AV_LOG_LEVEL"] = "quiet"
import site
import tempfile
import shutil
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

_npp_lib = Path(site.getsitepackages()[0]) / "nvidia" / "npp" / "lib"
_npp_so = _npp_lib / "libnppicc.so.12"
if _npp_so.is_file():
    ctypes.CDLL(str(_npp_so), mode=ctypes.RTLD_GLOBAL)

import torch
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = Path("./eval_results/vggsoundsync")

_openai_client = None

GPT_JUDGE_SYSTEM = """\
You are a structured-output extractor. The user will give you a model's free-text \
response about audio-video synchronization. Extract the following fields and return \
ONLY valid JSON (no markdown, no explanation):

{"synced": <bool>, "direction": "none"|"delay"|"early", "offset_sec": <float>}

Rules:
- synced: true if the model says audio and video are synchronized, false otherwise.
- direction: "delay" means audio comes AFTER the visual event; "early" means audio \
comes BEFORE the visual event; "none" if synced is true.
- offset_sec: estimated time gap in seconds. 0.0 if synced.
- If you cannot determine a field, use the default (true / "none" / 0.0).
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


def gpt_extract_prediction(raw_output: str, api_key: Optional[str] = None,
                           model: str = "gpt-5.4") -> Optional[Dict[str, Any]]:
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
                return {
                    "pred_synced": bool(synced),
                    "pred_direction": direction,
                    "pred_offset_sec": float(obj.get("offset_sec", 0.0)),
                    "parse_method": "gpt_judge",
                }
    except Exception as exc:
        print(f"  [gpt-judge] API error: {exc}", flush=True)
    return None


MCQ_PROMPT = """\
Watch this video and listen to its audio carefully.
Determine the synchronization status between the audio and video tracks.
Select the best answer:

A) The audio and video are synchronized.
B) The audio is delayed (comes after the visual event).
C) The audio is early (comes before the visual event).

Answer with only the letter (A, B, or C)."""

MCQ_PROMPT_SHUFFLED = """\
Watch this video and listen to its audio carefully.
Determine the synchronization status between the audio and video tracks.
Select the best answer:

A) The audio is early (comes before the visual event).
B) The audio and video are synchronized.
C) The audio is delayed (comes after the visual event).

Answer with only the letter (A, B, or C)."""

FREETEXT_PROMPT = """\
Watch this video and listen to its audio carefully. \
Determine whether the audio and video tracks are synchronized. \
If they are not synchronized, identify the direction of the offset \
(audio delayed or audio early relative to video) and estimate the offset in seconds. \
Explain your reasoning."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate on VGG-Sound Sync (out-of-domain sync).")
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--adapter", type=str, default=None)
    p.add_argument("--test-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--mode", choices=["mcq", "freetext"], default="mcq")
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--gpt-judge", action="store_true", default=False)
    p.add_argument("--openai-api-key", type=str, default=None)
    p.add_argument("--gpt-model", type=str, default="gpt-5.4")
    p.add_argument("--shuffle-mcq", action="store_true", default=False)
    p.add_argument("--vllm", action="store_true", default=False)
    p.add_argument("--tp", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=65536)
    return p.parse_args()


def load_test_data(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    base = path.parent.resolve()
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for key in ("video_path", "audio_path"):
                if key not in obj or not obj[key]:
                    continue
                p = Path(obj[key])
                if not p.is_absolute():
                    obj[key] = str((base / p).resolve())
            data.append(obj)
    if max_samples > 0:
        data = data[:max_samples]
    return data


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


def build_vllm_prompt(question: str, base_model: str, include_audio: bool = True) -> str:
    from omni_model_loading import vllm_user_mm_prefix

    mm = vllm_user_mm_prefix(base_model, include_audio=include_audio)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{mm}"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_model(base_model: str, adapter: Optional[str]):
    from omni_model_loading import load_qwen_omni_model

    model, processor, _ = load_qwen_omni_model(base_model, adapter)
    return model, processor


def run_inference(model, processor, video_path: str, audio_path: str,
                  prompt: str, max_new_tokens: int, temperature: float) -> str:
    from qwen_omni_utils import process_mm_info

    tmp_dir = tempfile.mkdtemp(prefix="eval_vggsync_")
    masked_video = os.path.join(tmp_dir, "clip.mp4")
    masked_audio = os.path.join(tmp_dir, "clip.wav")
    os.symlink(os.path.abspath(video_path), masked_video)
    os.symlink(os.path.abspath(audio_path), masked_audio)

    conversation = [{
        "role": "user",
        "content": [
            {"type": "video", "video": masked_video},
            {"type": "audio", "audio": masked_audio},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=True,
    )

    model_dtype = next(model.parameters()).dtype
    converted = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            v = v.to(model.device)
            if torch.is_floating_point(v):
                v = v.to(model_dtype)
        converted[k] = v
    inputs = converted

    from omni_model_loading import is_omni_thinker_model

    is_thinker = is_omni_thinker_model(model)
    if is_thinker:
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    else:
        gen_kwargs = {
            "thinker_max_new_tokens": max_new_tokens,
            "use_audio_in_video": True,
            "return_audio": False,
            "do_sample": temperature > 0,
        }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    if isinstance(output_ids, tuple):
        output_ids = output_ids[0]

    prompt_len = inputs["input_ids"].shape[1]
    response = processor.batch_decode(
        output_ids[:, prompt_len:], skip_special_tokens=True,
    )[0].strip()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return response


def extract_mcq_answer(text: str, answer_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    text = text.strip().upper()
    m = re.search(r"[ABC]", text)
    letter = m[0] if m else ""

    if answer_map is None:
        answer_map = {"A": "synced", "B": "delay", "C": "early"}

    key_to_pred = {
        "synced": {"pred_synced": True,  "pred_direction": "none"},
        "delay":  {"pred_synced": False, "pred_direction": "delay"},
        "early":  {"pred_synced": False, "pred_direction": "early"},
    }

    if letter in answer_map and answer_map[letter] in key_to_pred:
        return {**key_to_pred[answer_map[letter]], "pred_offset_sec": 0.0,
                "pred_letter": letter, "parse_method": "mcq"}
    return {"pred_synced": True, "pred_direction": "none", "pred_offset_sec": 0.0,
            "pred_letter": "", "parse_method": "mcq_failed"}


def extract_freetext_prediction(text: str) -> Dict[str, Any]:
    text_stripped = text.strip()
    for pattern in [
        re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL),
        re.compile(r"(\{[^{}]*\"synced\"[^{}]*\})", re.DOTALL),
        re.compile(r"(\{.*?\})", re.DOTALL),
    ]:
        m = pattern.search(text_stripped)
        if m:
            try:
                obj = json.loads(m.group(1))
                synced = obj.get("synced")
                if isinstance(synced, str):
                    synced = synced.lower() in ("true", "yes", "1")
                direction = str(obj.get("direction", "none")).lower().strip()
                if direction not in ("delay", "early", "none"):
                    direction = "none"
                return {
                    "pred_synced": bool(synced),
                    "pred_direction": direction,
                    "pred_offset_sec": float(obj.get("offset_sec", 0.0)),
                    "parse_method": "json",
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    text_lower = text_stripped.lower()
    synced = None
    direction = "none"
    offset = 0.0

    desync_kws = ["not synchronized", "not aligned", "desync", "mismatch",
                  "not in sync", "out of sync", "not well aligned"]
    sync_kws = ["synchronized", "well aligned", "well-aligned", "in sync",
                "closely aligned", "matches closely"]
    if any(kw in text_lower for kw in desync_kws):
        synced = False
    elif any(kw in text_lower for kw in sync_kws):
        synced = True

    if synced is False:
        delay_kws = ["audio delayed", "audio lags", "audio comes after",
                     "sound comes after", "audio is delayed", "sound follows"]
        early_kws = ["audio early", "audio leads", "audio comes before",
                     "sound comes before", "audio precedes", "audio is early"]
        if any(kw in text_lower for kw in delay_kws):
            direction = "delay"
        elif any(kw in text_lower for kw in early_kws):
            direction = "early"

        offset_match = re.search(
            r"(?:gap|offset|mismatch|differ\w*)\s*(?:of\s+)?(?:about\s+|roughly\s+|approximately\s+)?"
            r"([\d]+\.?\d*)\s*s", text_lower)
        if offset_match:
            offset = float(offset_match.group(1))

    if synced is None:
        synced = True

    return {
        "pred_synced": synced,
        "pred_direction": direction,
        "pred_offset_sec": offset,
        "parse_method": "regex_fallback",
    }


DIFFICULTY_ORDER = ["synced", "hard", "medium", "easy", "very_easy"]


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    sync_correct = sum(1 for r in results if r["pred_synced"] == r["gt_synced"])
    sync_acc = sync_correct / total

    def _label(r, prefix):
        if r[f"{prefix}synced"]:
            return "synced"
        return r[f"{prefix}direction"]

    three_class_correct = sum(1 for r in results if _label(r, "pred_") == _label(r, "gt_"))
    three_class_acc = three_class_correct / total

    desync = [r for r in results if not r["gt_synced"]]
    if desync:
        dir_correct = sum(1 for r in desync if r["pred_direction"] == r["gt_direction"])
        dir_acc = dir_correct / len(desync)
    else:
        dir_acc = None

    per_difficulty = {}
    for d in DIFFICULTY_ORDER:
        subset = [r for r in results if r["difficulty"] == d]
        if not subset:
            continue
        if d == "synced":
            acc = sum(1 for r in subset if r["pred_synced"]) / len(subset)
        else:
            acc = sum(1 for r in subset if _label(r, "pred_") == _label(r, "gt_")) / len(subset)
        per_difficulty[d] = {"accuracy": round(acc, 4), "count": len(subset)}

    per_class: Dict[str, Dict] = {}
    classes = sorted(set(r.get("label", "") for r in results))
    for cls in classes:
        subset = [r for r in results if r.get("label") == cls]
        if not subset:
            continue
        acc = sum(1 for r in subset if _label(r, "pred_") == _label(r, "gt_")) / len(subset)
        per_class[cls] = {"accuracy": round(acc, 4), "count": len(subset)}

    offset_errors = []
    for r in desync:
        if not r["pred_synced"] and r["pred_offset_sec"] > 0:
            offset_errors.append(abs(r["pred_offset_sec"] - r["gt_offset_sec"]))

    parse_stats = {}
    for r in results:
        m = r.get("parse_method", "unknown")
        parse_stats[m] = parse_stats.get(m, 0) + 1

    metrics = {
        "total_samples": total,
        "sync_desync_accuracy": round(sync_acc, 4),
        "three_class_accuracy": round(three_class_acc, 4),
        "direction_accuracy_on_desync": round(dir_acc, 4) if dir_acc is not None else None,
        "per_difficulty": per_difficulty,
        "per_class": per_class,
        "parse_stats": parse_stats,
    }
    if offset_errors:
        metrics["offset_mae_sec"] = round(mean(offset_errors), 4)
        metrics["offset_median_sec"] = round(median(offset_errors), 4)
        within_02 = sum(1 for e in offset_errors if e <= 0.2)
        within_05 = sum(1 for e in offset_errors if e <= 0.5)
        metrics["offset_within_0.2s"] = within_02
        metrics["offset_within_0.5s"] = within_05
        metrics["offset_evaluated_count"] = len(offset_errors)

    return metrics


def print_summary(metrics: Dict[str, Any], label: str) -> None:
    print()
    print(f"{'=' * 65}")
    print(f"  VGG-Sound Sync Eval: {label}")
    print(f"{'=' * 65}")
    print(f"  Total samples:         {metrics['total_samples']}")
    print(f"  Sync/Desync Accuracy:  {metrics['sync_desync_accuracy']:.1%}")
    print(f"  3-Class Accuracy:      {metrics['three_class_accuracy']:.1%}")
    if metrics.get("direction_accuracy_on_desync") is not None:
        print(f"  Direction Acc (desync): {metrics['direction_accuracy_on_desync']:.1%}")
    print(f"  ─── Per Difficulty ───")
    for d in DIFFICULTY_ORDER:
        if d in metrics.get("per_difficulty", {}):
            info = metrics["per_difficulty"][d]
            print(f"    {d:10s}: {info['accuracy']:.1%}  ({info['count']} samples)")
    if metrics.get("offset_mae_sec") is not None:
        print(f"  ─── Offset Estimation (freetext only) ───")
        print(f"    MAE:          {metrics['offset_mae_sec']:.3f}s")
        print(f"    Median Error: {metrics['offset_median_sec']:.3f}s")
        print(f"    Within 0.2s:  {metrics['offset_within_0.2s']} / {metrics['offset_evaluated_count']}")
        print(f"    Within 0.5s:  {metrics['offset_within_0.5s']} / {metrics['offset_evaluated_count']}")
    print(f"  ─── Parse Stats ───")
    for method, count in sorted(metrics.get("parse_stats", {}).items()):
        print(f"    {method}: {count}")
    print(f"{'=' * 65}")


def _extract_pred(raw_output: str, mode: str, gpt_judge: bool,
                   openai_api_key: Optional[str], gpt_model: str,
                   answer_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    if mode == "mcq":
        return extract_mcq_answer(raw_output, answer_map=answer_map)
    if gpt_judge and raw_output:
        gpt_pred = gpt_extract_prediction(raw_output, api_key=openai_api_key, model=gpt_model)
        if gpt_pred is not None:
            return gpt_pred
    return extract_freetext_prediction(raw_output)


def _build_result(item: Dict, pred: Dict, raw_output: str, mode: str) -> Dict:
    result = {
        "uid": item["uid"],
        "ytid": item["ytid"],
        "label": item.get("label", ""),
        "difficulty": item["difficulty"],
        "gt_synced": item["gt_synced"],
        "gt_direction": item["gt_direction"],
        "gt_offset_sec": item["gt_offset_sec"],
        "pred_synced": pred["pred_synced"],
        "pred_direction": pred["pred_direction"],
        "pred_offset_sec": pred.get("pred_offset_sec", 0.0),
        "parse_method": pred["parse_method"],
        "raw_output": raw_output,
    }
    if mode == "mcq":
        result["pred_letter"] = pred.get("pred_letter", "")
    return result


def _save_and_finalize(results_jsonl: Path, metrics_json: Path, summary_txt: Path,
                       args, label: str):
    all_results = []
    if results_jsonl.exists():
        with open(results_jsonl) as f:
            for line in f:
                all_results.append(json.loads(line))

    if not all_results:
        print("[warn] No results.")
        return

    metrics = compute_metrics(all_results)
    metrics["eval_config"] = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "mode": args.mode,
        "test_jsonl": str(args.test_jsonl),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
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

    print(f"\n[output] Results: {results_jsonl}")
    print(f"[output] Metrics: {metrics_json}")
    print(f"[output] Summary: {summary_txt}")


def main() -> None:
    args = parse_args()
    label = args.label or (Path(args.adapter).name if args.adapter else Path(args.base_model).name)
    default_prompt = MCQ_PROMPT if args.mode == "mcq" else FREETEXT_PROMPT

    if args.gpt_judge and args.mode == "freetext":
        client = _get_openai_client(args.openai_api_key)
        if client is None:
            print("[ERROR] --gpt-judge requires OPENAI_API_KEY or --openai-api-key.")
            raise SystemExit(1)

    out_dir = args.output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = out_dir / "eval_results.jsonl"
    metrics_json = out_dir / "metrics.json"
    summary_txt = out_dir / "summary.txt"

    test_data = load_test_data(args.test_jsonl, args.max_samples)
    print(f"[data] {len(test_data)} samples loaded (mode={args.mode})")

    processed = set()
    if results_jsonl.exists():
        with open(results_jsonl) as f:
            for line in f:
                processed.add(json.loads(line)["uid"])
        print(f"[resume] {len(processed)} already done")

    use_vllm = args.vllm

    if use_vllm:
        from vllm import LLM, SamplingParams

        tp = args.tp or torch.cuda.device_count()
        todo = [item for item in test_data if item["uid"] not in processed]

        uniq_v = list(dict.fromkeys(item["video_path"] for item in todo))
        uniq_a = list(dict.fromkeys(item["audio_path"] for item in todo))
        print(
            f"[vllm] Phase 1 — CPU preprocess: {len(uniq_v)} unique videos, {len(uniq_a)} unique audios "
            f"for {len(todo)} samples (GPUs idle until model load).",
            flush=True,
        )
        preprocessed_v: Dict[str, Any] = {}
        preprocessed_a: Dict[str, Any] = {}
        failed_paths: set = set()

        for vp in tqdm(uniq_v, desc="Preprocess video", unit="file"):
            if vp in failed_paths:
                continue
            try:
                preprocessed_v[vp] = preprocess_video_for_vllm(vp)
            except Exception as e:
                failed_paths.add(vp)
                print(f"  [skip] video preprocess error: {Path(vp).name}: {e}")

        for ap in tqdm(uniq_a, desc="Preprocess audio", unit="file"):
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
        print(f"[vllm] {len(vllm_todo)} samples ready, {len(fallback_items)} deferred to fallback ...")

        for i, item in enumerate(vllm_todo):
            if item["uid"] in processed:
                continue
            item_prompt = item.get("mcq_prompt", default_prompt) if args.mode == "mcq" else default_prompt
            item_answer_map = item.get("mcq_answer_map") if args.mode == "mcq" else None
            inp = {
                "prompt": build_vllm_prompt(item_prompt, args.base_model, include_audio=True),
                "multi_modal_data": {
                    "video": preprocessed_v[item["video_path"]],
                    "audio": preprocessed_a[item["audio_path"]],
                },
            }
            try:
                outputs = llm.generate([inp], sampling_params=sampling_params)
                raw_output = outputs[0].outputs[0].text.strip()
            except (ValueError, RuntimeError) as exc:
                print(f"  [error] {item['uid']}: {exc}")
                raw_output = ""

            pred = _extract_pred(raw_output, args.mode, args.gpt_judge,
                                 args.openai_api_key, args.gpt_model,
                                 answer_map=item_answer_map)
            result = _build_result(item, pred, raw_output, args.mode)

            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed.add(item["uid"])

            if (i + 1) % 100 == 0:
                print(f"  [vllm] [{i + 1}/{len(vllm_todo)}] done")

        preprocessed_v.clear()
        preprocessed_a.clear()

        if fallback_items:
            print(f"[fallback] Running {len(fallback_items)} samples with transformers ...")
            del llm
            gc.collect()
            torch.cuda.empty_cache()

            model, processor = load_model(args.base_model, args.adapter)
            for item in tqdm(fallback_items, desc="Fallback", unit="q"):
                if item["uid"] in processed:
                    continue
                item_prompt = item.get("mcq_prompt", default_prompt) if args.mode == "mcq" else default_prompt
                item_answer_map = item.get("mcq_answer_map") if args.mode == "mcq" else None
                try:
                    raw_output = run_inference(
                        model, processor, item["video_path"], item["audio_path"],
                        item_prompt, args.max_new_tokens, args.temperature,
                    )
                except Exception as exc:
                    import traceback
                    print(f"  [error] {item['uid']}: {exc}")
                    traceback.print_exc()
                    raw_output = ""

                pred = _extract_pred(raw_output, args.mode, args.gpt_judge,
                                     args.openai_api_key, args.gpt_model,
                                     answer_map=item_answer_map)
                result = _build_result(item, pred, raw_output, args.mode)

                with open(results_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed.add(item["uid"])
                gc.collect()
                torch.cuda.empty_cache()

    else:
        model, processor = load_model(args.base_model, args.adapter)

        for item in tqdm(test_data, desc="VGGSync", unit="q"):
            if item["uid"] in processed:
                continue
            if not os.path.exists(item["video_path"]):
                print(f"  [skip] video not found: {item['video_path']}")
                continue
            if not os.path.exists(item["audio_path"]):
                print(f"  [skip] audio not found: {item['audio_path']}")
                continue

            item_prompt = item.get("mcq_prompt", default_prompt) if args.mode == "mcq" else default_prompt
            item_answer_map = item.get("mcq_answer_map") if args.mode == "mcq" else None

            try:
                raw_output = run_inference(
                    model, processor, item["video_path"], item["audio_path"],
                    item_prompt, args.max_new_tokens, args.temperature,
                )
            except Exception as exc:
                import traceback
                print(f"  [error] {item['uid']}: {exc}")
                traceback.print_exc()
                raw_output = ""

            pred = _extract_pred(raw_output, args.mode, args.gpt_judge,
                                 args.openai_api_key, args.gpt_model,
                                 answer_map=item_answer_map)
            result = _build_result(item, pred, raw_output, args.mode)

            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed.add(item["uid"])
            gc.collect()
            torch.cuda.empty_cache()

    _save_and_finalize(results_jsonl, metrics_json, summary_txt, args, label)


if __name__ == "__main__":
    main()
