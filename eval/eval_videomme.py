#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import site
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

_npp_lib = Path(site.getsitepackages()[0]) / "nvidia" / "npp" / "lib"
_npp_so = _npp_lib / "libnppicc.so.12"
if _npp_so.is_file():
    ctypes.CDLL(str(_npp_so), mode=ctypes.RTLD_GLOBAL)

import torch
from tqdm import tqdm

DEFAULT_VIDEO_DIR = Path("./data/videomme/data")
DEFAULT_OUTPUT_DIR = Path("./eval_results/videomme")

VIDEO_TYPES = ["short", "medium", "long"]
CATEGORIES = [
    "Knowledge", "Film & Television", "Sports Competition",
    "Artistic Performance", "Life Record", "Multilingual",
]
TASK_CATEGORIES = [
    "Temporal Perception", "Spatial Perception", "Attribute Perception",
    "Action Recognition", "Object Recognition", "OCR Problems",
    "Counting Problem", "Temporal Reasoning", "Spatial Reasoning",
    "Action Reasoning", "Object Reasoning", "Information Synopsis",
]

MCQ_PROMPT = (
    "Select the best answer to the following multiple-choice question "
    "based on the video. Respond with only the letter (A, B, C, or D) "
    "of the correct option.\n"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate on Video-MME benchmark.")
    p.add_argument("--base-model", type=str,
                    default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--adapter", type=str, default=None)
    p.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--vllm", action="store_true", default=False)
    p.add_argument("--tp", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=65536)
    return p.parse_args()


def load_model(base_model: str, adapter: Optional[str]):
    from omni_model_loading import load_qwen_omni_model

    model, processor, _ = load_qwen_omni_model(base_model, adapter)
    return model, processor


def run_inference(model, processor, video_path: str, prompt: str,
                  max_new_tokens: int, temperature: float) -> str:
    from qwen_omni_utils import process_mm_info

    tmp_dir = tempfile.mkdtemp(prefix="eval_vmme_")
    masked_video = os.path.join(tmp_dir, "clip.mp4")
    os.symlink(os.path.abspath(video_path), masked_video)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": masked_video},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
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


def preprocess_video_for_vllm(video_path: str):
    from qwen_omni_utils import process_mm_info
    import numpy as np

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "nframes": 128},
            {"type": "text", "text": "placeholder"},
        ],
    }]
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    video_tensor = videos[0]
    video_np = (video_tensor * 255).byte().numpy()
    audio_tuple = None
    if audios:
        aud = audios[0]
        if isinstance(aud, tuple):
            audio_tuple = (aud[0].numpy() if hasattr(aud[0], "numpy") else np.asarray(aud[0]),
                           aud[1])
        elif hasattr(aud, "numpy"):
            audio_tuple = (aud.numpy(), 16000)
        else:
            audio_tuple = (np.asarray(aud), 16000)
    return video_np, audio_tuple


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


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


def extract_answer(text: str) -> str:
    text = text.strip()
    prefixes = [
        "The best answer is", "The correct answer is",
        "The answer is", "The answer", "Best answer:", "Best option:",
    ]
    for prefix in prefixes:
        text = text.replace(prefix, "")

    if len(text.split()) > 10 and not re.search(r"[ABCD]", text):
        return ""
    m = re.search(r"[ABCD]", text)
    return m[0] if m else ""


def load_videomme(video_dir: Path, max_samples: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/Video-MME", split="test")
    data = []
    skipped = 0
    for row in ds:
        vid = row["videoID"]
        video_path = video_dir / f"{vid}.mp4"
        if not video_path.exists():
            for ext in [".MP4", ".mkv"]:
                alt = video_dir / f"{vid}{ext}"
                if alt.exists():
                    video_path = alt
                    break
        if not video_path.exists():
            skipped += 1
            continue

        options_text = "\n".join(row["options"])
        prompt = MCQ_PROMPT + row["question"] + "\n" + options_text + "\nThe best answer is:"

        data.append({
            "question_id": row["question_id"],
            "video_id": vid,
            "video_path": str(video_path),
            "duration": row["duration"],
            "domain": row["domain"],
            "sub_category": row["sub_category"],
            "task_type": row["task_type"],
            "question": row["question"],
            "options": row["options"],
            "gt_answer": row["answer"],
            "prompt": prompt,
        })
    if skipped:
        print(f"[data] Skipped {skipped} questions (video not found)")
    if max_samples > 0:
        data = data[:max_samples]
    return data


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    correct = sum(1 for r in results if r["pred_answer"].upper() == r["gt_answer"].upper())
    overall_acc = correct / total

    def acc_for(items):
        if not items:
            return None
        c = sum(1 for r in items if r["pred_answer"].upper() == r["gt_answer"].upper())
        return round(c / len(items), 4)

    per_duration = {}
    for vt in VIDEO_TYPES:
        subset = [r for r in results if r["duration"] == vt]
        if subset:
            per_duration[vt] = {"accuracy": acc_for(subset), "count": len(subset)}

    per_domain = {}
    for cat in CATEGORIES:
        subset = [r for r in results if r["domain"] == cat]
        if subset:
            per_domain[cat] = {"accuracy": acc_for(subset), "count": len(subset)}

    per_task = {}
    for task in TASK_CATEGORIES:
        subset = [r for r in results if r["task_type"] == task]
        if subset:
            per_task[task] = {"accuracy": acc_for(subset), "count": len(subset)}

    return {
        "total_samples": total,
        "overall_accuracy": round(overall_acc, 4),
        "per_duration": per_duration,
        "per_domain": per_domain,
        "per_task_type": per_task,
    }


def print_summary(metrics: Dict[str, Any], label: str) -> None:
    print()
    print(f"{'=' * 65}")
    print(f"  Video-MME Summary: {label}")
    print(f"{'=' * 65}")
    print(f"  Total samples:         {metrics['total_samples']}")
    print(f"  Overall Accuracy:      {metrics['overall_accuracy']:.1%}")

    print(f"  ─── Per Duration ───")
    for vt in VIDEO_TYPES:
        if vt in metrics["per_duration"]:
            d = metrics["per_duration"][vt]
            print(f"    {vt:8s}:  {d['accuracy']:.1%}  ({d['count']} questions)")

    print(f"  ─── Per Domain ───")
    for cat in CATEGORIES:
        if cat in metrics["per_domain"]:
            d = metrics["per_domain"][cat]
            print(f"    {cat:25s}: {d['accuracy']:.1%}  ({d['count']})")

    print(f"  ─── Per Task Type ───")
    for task in TASK_CATEGORIES:
        if task in metrics["per_task_type"]:
            d = metrics["per_task_type"][task]
            print(f"    {task:25s}: {d['accuracy']:.1%}  ({d['count']})")

    print(f"{'=' * 65}")


def main() -> None:
    args = parse_args()
    label = args.label or (
        Path(args.adapter).name if args.adapter
        else Path(args.base_model).name
    )

    out_dir = args.output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = out_dir / "eval_results.jsonl"
    metrics_json = out_dir / "metrics.json"
    summary_txt = out_dir / "summary.txt"

    print("[data] Loading Video-MME dataset...")
    test_data = load_videomme(args.video_dir, args.max_samples)
    print(f"[data] {len(test_data)} questions ready for evaluation")

    processed = set()
    if results_jsonl.exists():
        with open(results_jsonl) as f:
            for line in f:
                obj = json.loads(line)
                processed.add(obj["question_id"])
        print(f"[resume] {len(processed)} already processed, skipping")

    use_vllm = args.vllm
    model = processor = llm = None
    vllm_preprocess_stats: Dict[str, int] | None = None

    if use_vllm:
        from vllm import LLM, SamplingParams
        tp = args.tp or torch.cuda.device_count()
        model_path = args.base_model

        print("[vllm] Preprocessing videos (before model load) ...")
        todo = [item for item in test_data if item["question_id"] not in processed]
        unique_videos = list(dict.fromkeys(item["video_path"] for item in todo))
        from omni_model_loading import parallel_preprocess_videos
        preprocessed, preprocessed_audio, preprocess_failed_paths = parallel_preprocess_videos(
            unique_videos, preprocess_video_for_vllm,
        )

        n_pp_skip = sum(1 for item in todo if item["video_path"] in preprocess_failed_paths)
        if preprocess_failed_paths:
            print(
                f"[vllm] Preprocess failed for {len(preprocess_failed_paths)} video(s), "
                f"{n_pp_skip} question(s) will not use vLLM (run continues)."
            )
        vllm_preprocess_stats = {
            "preprocess_failed_videos": len(preprocess_failed_paths),
            "preprocess_skipped_questions": n_pp_skip,
        }

        from omni_model_loading import cap_vllm_max_model_len

        vllm_max_len = cap_vllm_max_model_len(model_path, args.max_model_len)
        print(f"[vllm] Loading {model_path} with tp={tp} (max_model_len={vllm_max_len}) ...")
        llm = LLM(
            model=model_path,
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

        vllm_todo = [item for item in todo if item["video_path"] in preprocessed]
        fallback_items = []
        print(f"[vllm] {len(vllm_todo)} questions ready, running inference ...")

        for i, item in enumerate(vllm_todo):
            if item["question_id"] in processed:
                continue
            inp = {
                "prompt": build_vllm_prompt(item["prompt"], args.base_model),
                "multi_modal_data": {"video": preprocessed[item["video_path"]]},
            }
            if item["video_path"] in preprocessed_audio:
                inp["multi_modal_data"]["audio"] = preprocessed_audio[item["video_path"]]
            try:
                outputs = llm.generate([inp], sampling_params=sampling_params)
                raw_output = outputs[0].outputs[0].text.strip()
                pred = extract_answer(raw_output)
                result = {
                    "question_id": item["question_id"],
                    "video_id": item["video_id"],
                    "duration": item["duration"],
                    "domain": item["domain"],
                    "sub_category": item["sub_category"],
                    "task_type": item["task_type"],
                    "gt_answer": item["gt_answer"],
                    "pred_answer": pred,
                    "correct": pred.upper() == item["gt_answer"].upper(),
                    "raw_output": raw_output,
                }
                with open(results_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed.add(item["question_id"])
            except (ValueError, RuntimeError) as exc:
                if "longer than the maximum model length" in str(exc):
                    print(f"  [too long] {item['question_id']} -> fallback")
                    fallback_items.append(item)
                else:
                    raise

            if (i + 1) % 100 == 0:
                print(f"  [vllm] [{i+1}/{len(vllm_todo)}] done, {len(fallback_items)} deferred")

        preprocessed.clear()
        preprocessed_audio.clear()

        vllm_results = []
        if results_jsonl.exists():
            with open(results_jsonl) as f:
                for line in f:
                    vllm_results.append(json.loads(line))
        if vllm_results:
            vllm_metrics = compute_metrics(vllm_results)
            vllm_metrics_path = out_dir / "metrics_vllm.json"
            with open(vllm_metrics_path, "w", encoding="utf-8") as f:
                json.dump(vllm_metrics, f, indent=2, ensure_ascii=False)
            print(f"[vllm] Intermediate metrics saved to {vllm_metrics_path}")
            print_summary(vllm_metrics, label + " (vllm only)")

        if fallback_items:
            print(f"[fallback] Running {len(fallback_items)} long-video questions with transformers ...")
            del llm
            gc.collect()
            torch.cuda.empty_cache()

            model, processor = load_model(args.base_model, args.adapter)
            for item in tqdm(fallback_items, desc="Fallback", unit="q"):
                if item["question_id"] in processed:
                    continue
                try:
                    raw_output = run_inference(
                        model, processor, item["video_path"], item["prompt"],
                        args.max_new_tokens, args.temperature,
                    )
                except Exception as exc:
                    import traceback
                    print(f"  [error] {item['question_id']}: {exc}")
                    traceback.print_exc()
                    raw_output = ""

                pred = extract_answer(raw_output)
                result = {
                    "question_id": item["question_id"],
                    "video_id": item["video_id"],
                    "duration": item["duration"],
                    "domain": item["domain"],
                    "sub_category": item["sub_category"],
                    "task_type": item["task_type"],
                    "gt_answer": item["gt_answer"],
                    "pred_answer": pred,
                    "correct": pred.upper() == item["gt_answer"].upper(),
                    "raw_output": raw_output,
                }
                with open(results_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed.add(item["question_id"])
                gc.collect()
                torch.cuda.empty_cache()

    else:
        print("[model] Loading model...")
        model, processor = load_model(args.base_model, args.adapter)

        for item in tqdm(test_data, desc="Video-MME", unit="q"):
            if item["question_id"] in processed:
                continue

            try:
                raw_output = run_inference(
                    model, processor, item["video_path"], item["prompt"],
                    args.max_new_tokens, args.temperature,
                )
            except Exception as exc:
                import traceback
                print(f"  [error] {item['question_id']}: {exc}")
                traceback.print_exc()
                raw_output = ""

            pred = extract_answer(raw_output)

            result = {
                "question_id": item["question_id"],
                "video_id": item["video_id"],
                "duration": item["duration"],
                "domain": item["domain"],
                "sub_category": item["sub_category"],
                "task_type": item["task_type"],
                "gt_answer": item["gt_answer"],
                "pred_answer": pred,
                "correct": pred.upper() == item["gt_answer"].upper(),
                "raw_output": raw_output,
            }

            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            processed.add(item["question_id"])
            gc.collect()
            torch.cuda.empty_cache()

    all_results = []
    if results_jsonl.exists():
        with open(results_jsonl) as f:
            for line in f:
                all_results.append(json.loads(line))

    if not all_results:
        print("[warn] No results to compute metrics from.")
        return

    metrics = compute_metrics(all_results)
    metrics["eval_config"] = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "video_dir": str(args.video_dir),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    if vllm_preprocess_stats is not None:
        metrics["eval_config"]["vllm_preprocess_skips"] = vllm_preprocess_stats

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


if __name__ == "__main__":
    main()
