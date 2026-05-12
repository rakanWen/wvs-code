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

DEFAULT_VIDEO_DIR = Path("./data/lvbench")
DEFAULT_OUTPUT_DIR = Path("./eval_results/lvbench")

VIDEO_TYPES = ["cartoon", "documentary", "live", "selfmedia", "sport", "tv"]

MCQ_PROMPT = (
    "Select the best answer to the following multiple-choice question "
    "based on the video. Respond with only the letter (A, B, C, or D) "
    "of the correct option.\n"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate on LVBench benchmark.")
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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--vllm-batch-size", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true", default=False)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--merge-only", action="store_true", default=False)
    return p.parse_args()


def load_model(base_model: str, adapter: Optional[str]):
    from omni_model_loading import load_qwen_omni_model

    model, processor, _ = load_qwen_omni_model(base_model, adapter)
    return model, processor


def run_inference(model, processor, video_path: str, prompt: str,
                  max_new_tokens: int, temperature: float,
                  cached_mm: Optional[Dict[str, Any]] = None) -> str:
    from qwen_omni_utils import process_mm_info

    tmp_dir = tempfile.mkdtemp(prefix="eval_lvb_")
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
    if cached_mm is not None:
        audios, images, videos = cached_mm["audios"], cached_mm["images"], cached_mm["videos"]
    else:
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
            {"type": "video", "video": video_path, "fps": 2.0, "max_frames": 128},
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


def load_lvbench(video_dir: Path, max_samples: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/LVBench", split="train")
    data = []
    skipped = 0
    for row in ds:
        vid = row["key"]
        video_path = video_dir / f"{vid}.mp4"
        if not video_path.exists():
            skipped += 1
            continue

        prompt = MCQ_PROMPT + row["question"] + "\nThe best answer is:"

        data.append({
            "uid": row["uid"],
            "video_id": vid,
            "video_path": str(video_path),
            "video_type": row["type"],
            "question_type": row["question_type"],
            "question": row["question"],
            "gt_answer": row["answer"],
            "time_reference": row.get("time_reference", ""),
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

    per_type = {}
    for vt in VIDEO_TYPES:
        subset = [r for r in results if r["video_type"] == vt]
        if subset:
            per_type[vt] = {"accuracy": acc_for(subset), "count": len(subset)}

    q_types = set()
    for r in results:
        if isinstance(r.get("question_type"), list):
            q_types.update(r["question_type"])
        elif r.get("question_type"):
            q_types.add(r["question_type"])

    per_qtype = {}
    for qt in sorted(q_types):
        subset = [r for r in results if qt in (r.get("question_type", [])
                  if isinstance(r.get("question_type"), list) else [r.get("question_type")])]
        if subset:
            per_qtype[qt] = {"accuracy": acc_for(subset), "count": len(subset)}

    return {
        "total_samples": total,
        "overall_accuracy": round(overall_acc, 4),
        "per_video_type": per_type,
        "per_question_type": per_qtype,
    }


def print_summary(metrics: Dict[str, Any], label: str) -> None:
    print()
    print(f"{'=' * 65}")
    print(f"  LVBench Summary: {label}")
    print(f"{'=' * 65}")
    print(f"  Total samples:         {metrics['total_samples']}")
    print(f"  Overall Accuracy:      {metrics['overall_accuracy']:.1%}")

    print(f"  ─── Per Video Type ───")
    for vt in VIDEO_TYPES:
        if vt in metrics.get("per_video_type", {}):
            d = metrics["per_video_type"][vt]
            print(f"    {vt:15s}: {d['accuracy']:.1%}  ({d['count']} questions)")

    print(f"  ─── Per Question Type ───")
    for qt, d in sorted(metrics.get("per_question_type", {}).items()):
        print(f"    {qt:30s}: {d['accuracy']:.1%}  ({d['count']})")

    print(f"{'=' * 65}")


def _load_processed_uids(out_dir: Path) -> set:
    processed: set = set()
    for p in sorted(out_dir.glob("eval_results*.jsonl")):
        try:
            with open(p) as f:
                for line in f:
                    try:
                        processed.add(json.loads(line)["uid"])
                    except Exception:
                        continue
        except FileNotFoundError:
            continue
    return processed


def _finalize_metrics(out_dir: Path, label: str, args: argparse.Namespace,
                      vllm_preprocess_stats: Optional[Dict[str, int]] = None) -> None:
    results_by_uid: Dict[str, Dict[str, Any]] = {}
    source_files = sorted(out_dir.glob("eval_results*.jsonl"))
    for p in source_files:
        with open(p) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                results_by_uid[obj["uid"]] = obj

    if not results_by_uid:
        print("[warn] No results to compute metrics from.")
        return

    all_results = list(results_by_uid.values())
    merged_jsonl = out_dir / "eval_results.jsonl"
    with open(merged_jsonl, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] Wrote {len(all_results)} unique results to {merged_jsonl} "
          f"(merged {len(source_files)} source file(s)).")

    metrics = compute_metrics(all_results)
    metrics["eval_config"] = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "video_dir": str(args.video_dir),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "vllm": bool(args.vllm),
        "max_num_seqs": args.max_num_seqs,
        "vllm_batch_size": args.vllm_batch_size,
        "max_model_len": args.max_model_len,
        "num_shards": args.num_shards,
    }
    if vllm_preprocess_stats is not None:
        metrics["eval_config"]["vllm_preprocess_skips"] = vllm_preprocess_stats

    metrics_json = out_dir / "metrics.json"
    summary_txt = out_dir / "summary.txt"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print_summary(metrics, label)

    with open(summary_txt, "w", encoding="utf-8") as f:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_summary(metrics, label)
        f.write(buf.getvalue())

    print(f"\n[output] Results: {merged_jsonl}")
    print(f"[output] Metrics: {metrics_json}")
    print(f"[output] Summary: {summary_txt}")


def main() -> None:
    args = parse_args()
    label = args.label or (
        Path(args.adapter).name if args.adapter
        else Path(args.base_model).name
    )

    out_dir = args.output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.num_shards < 1 or args.shard < 0 or args.shard >= args.num_shards:
        raise SystemExit(f"Invalid --shard {args.shard} / --num-shards {args.num_shards}")
    is_sharded = args.num_shards > 1
    shard_tag = f".shard{args.shard}of{args.num_shards}" if is_sharded else ""
    results_jsonl = out_dir / f"eval_results{shard_tag}.jsonl"

    if args.merge_only:
        print(f"[merge-only] out_dir={out_dir}")
        _finalize_metrics(out_dir, label, args)
        return

    print("[data] Loading LVBench dataset...")
    test_data = load_lvbench(args.video_dir, args.max_samples)
    print(f"[data] {len(test_data)} total questions")

    if is_sharded:
        shard_data = [x for i, x in enumerate(test_data)
                      if i % args.num_shards == args.shard]
        print(f"[shard] shard={args.shard}/{args.num_shards} -> "
              f"{len(shard_data)} questions in this shard")
        test_data = shard_data

    processed = _load_processed_uids(out_dir)
    if processed:
        print(f"[resume] {len(processed)} uids already processed across all "
              f"eval_results*.jsonl under {out_dir}")

    use_vllm = args.vllm
    model = processor = llm = None
    vllm_preprocess_stats: Dict[str, int] | None = None

    if use_vllm:
        from vllm import LLM, SamplingParams
        tp = args.tp or torch.cuda.device_count()
        model_path = args.base_model

        print(f"[vllm] Preprocessing videos (before model load) ...")
        todo = [item for item in test_data if item["uid"] not in processed]
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
        print(f"[vllm] Loading {model_path} with tp={tp} "
              f"(max_num_seqs={args.max_num_seqs}, max_model_len={vllm_max_len}) ...")
        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=tp,
            max_model_len=vllm_max_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
            limit_mm_per_prompt={"video": 1, "audio": 1},
            enforce_eager=args.enforce_eager,
        )
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=args.temperature if args.temperature > 0 else 0.0,
            top_p=0.9 if args.temperature > 0 else 1.0,
            max_tokens=args.max_new_tokens,
        )

        vllm_todo = [item for item in todo if item["video_path"] in preprocessed]
        fallback_items = []
        print(f"[vllm] {len(vllm_todo)} questions ready, running inference "
              f"(batch={args.vllm_batch_size}) ...")

        def _write_result(item: Dict[str, Any], raw_output: str) -> None:
            pred = extract_answer(raw_output)
            result = {
                "uid": item["uid"],
                "video_id": item["video_id"],
                "video_type": item["video_type"],
                "question_type": item["question_type"],
                "gt_answer": item["gt_answer"],
                "pred_answer": pred,
                "correct": pred.upper() == item["gt_answer"].upper(),
                "raw_output": raw_output,
            }
            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed.add(item["uid"])

        def _build_inp(item: Dict[str, Any]) -> Dict[str, Any]:
            inp = {
                "prompt": build_vllm_prompt(item["prompt"], args.base_model),
                "multi_modal_data": {"video": preprocessed[item["video_path"]]},
            }
            if item["video_path"] in preprocessed_audio:
                inp["multi_modal_data"]["audio"] = preprocessed_audio[item["video_path"]]
            return inp

        def _flush(batch: List[Dict[str, Any]]) -> None:
            if not batch:
                return
            inps = [b["inp"] for b in batch]
            try:
                outs = llm.generate(inps, sampling_params=sampling_params)
                for b, o in zip(batch, outs):
                    _write_result(b["item"], o.outputs[0].text.strip())
                return
            except (ValueError, RuntimeError) as exc:
                if "longer than the maximum model length" not in str(exc):
                    raise
            for b in batch:
                try:
                    outs = llm.generate([b["inp"]], sampling_params=sampling_params)
                    _write_result(b["item"], outs[0].outputs[0].text.strip())
                except (ValueError, RuntimeError) as exc2:
                    if "longer than the maximum model length" in str(exc2):
                        print(f"  [too long] {b['item']['uid']} -> fallback")
                        fallback_items.append(b["item"])
                    else:
                        raise

        batch: List[Dict[str, Any]] = []
        for i, item in enumerate(vllm_todo):
            if item["uid"] in processed:
                continue
            batch.append({"inp": _build_inp(item), "item": item})
            if len(batch) >= max(1, args.vllm_batch_size):
                _flush(batch)
                batch = []
            if (i + 1) % 50 == 0:
                print(f"  [vllm] [{i+1}/{len(vllm_todo)}] submitted, "
                      f"{len(fallback_items)} deferred")
        _flush(batch)

        preprocessed.clear()
        preprocessed_audio.clear()

        if fallback_items:
            print(f"[fallback] Running {len(fallback_items)} long-video questions with transformers ...")
            del llm
            gc.collect()
            torch.cuda.empty_cache()

            fallback_items.sort(key=lambda it: it["video_path"])

            model, processor = load_model(args.base_model, args.adapter)
            last_vp: Optional[str] = None
            cached_mm: Optional[Dict[str, Any]] = None
            for item in tqdm(fallback_items, desc="Fallback", unit="q"):
                if item["uid"] in processed:
                    continue
                if item["video_path"] != last_vp:
                    cached_mm = None
                    last_vp = item["video_path"]
                try:
                    if cached_mm is None:
                        from qwen_omni_utils import process_mm_info as _pmi
                        tmp_conv = [{"role": "user", "content": [
                            {"type": "video", "video": item["video_path"]},
                            {"type": "text", "text": item["prompt"]},
                        ]}]
                        a, im, v = _pmi(tmp_conv, use_audio_in_video=True)
                        cached_mm = {"audios": a, "images": im, "videos": v}
                    raw_output = run_inference(
                        model, processor, item["video_path"], item["prompt"],
                        args.max_new_tokens, args.temperature,
                        cached_mm=cached_mm,
                    )
                except Exception as exc:
                    import traceback
                    print(f"  [error] {item['uid']}: {exc}")
                    traceback.print_exc()
                    raw_output = ""
                    cached_mm = None

                pred = extract_answer(raw_output)
                result = {
                    "uid": item["uid"],
                    "video_id": item["video_id"],
                    "video_type": item["video_type"],
                    "question_type": item["question_type"],
                    "gt_answer": item["gt_answer"],
                    "pred_answer": pred,
                    "correct": pred.upper() == item["gt_answer"].upper(),
                    "raw_output": raw_output,
                }
                with open(results_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed.add(item["uid"])
                gc.collect()
                torch.cuda.empty_cache()

    else:
        print("[model] Loading model...")
        model, processor = load_model(args.base_model, args.adapter)

        for item in tqdm(test_data, desc="LVBench", unit="q"):
            if item["uid"] in processed:
                continue

            try:
                raw_output = run_inference(
                    model, processor, item["video_path"], item["prompt"],
                    args.max_new_tokens, args.temperature,
                )
            except Exception as exc:
                import traceback
                print(f"  [error] {item['uid']}: {exc}")
                traceback.print_exc()
                raw_output = ""

            pred = extract_answer(raw_output)

            result = {
                "uid": item["uid"],
                "video_id": item["video_id"],
                "video_type": item["video_type"],
                "question_type": item["question_type"],
                "gt_answer": item["gt_answer"],
                "pred_answer": pred,
                "correct": pred.upper() == item["gt_answer"].upper(),
                "raw_output": raw_output,
            }

            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            processed.add(item["uid"])
            gc.collect()
            torch.cuda.empty_cache()

    if is_sharded:
        print(f"[shard {args.shard}/{args.num_shards}] finished inference. "
              f"Run `--merge-only` after all shards complete to produce final metrics.")
        return

    _finalize_metrics(out_dir, label, args, vllm_preprocess_stats)


if __name__ == "__main__":
    main()
