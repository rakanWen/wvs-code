
## 🚀 Official Codebase

This is the official code repository for our paper **When Vision Speaks for Sound**.

It provides the code, model release, and evaluation interface for **THUD**, an intervention-driven diagnostic framework for probing whether video-capable multimodal models truly verify audio or rely on visual-semantic shortcuts.

---
### ⚙️ Environment Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Some system-level dependencies are not included in `requirements.txt`.  
For video/audio processing and DeepSpeed compilation, please also make sure that `ffmpeg`, CUDA toolkit / `nvcc`, and the required NVIDIA libraries are available in your environment.

We use **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** for SFT and DPO training. Please install LLaMA-Factory separately following its official instructions, or clone it manually:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```


### 🔧 Training with LLaMA-Factory

To reproduce or adapt the training process, please first register the corresponding datasets in:

```bash
LLaMA-Factory/data/data_info.json
```

---

#### SFT Data Format

The SFT data follows the ShareGPT-style multimodal format. Each example contains a `messages` field, together with the corresponding video and audio paths:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<video><audio>Is there any noticeable audio delay or temporal manipulation in this clip?"
    },
    {
      "role": "assistant",
      "content": "The moment a child running with a blanket over their head collides with a pile of toys and falls lines up well with the thud and clatter of plastic toys, so this clip appears synchronized overall."
    }
  ],
  "videos": [
    "/path/to/video.mp4"
  ],
  "audios": [
    "/path/to/audio.wav"
  ]
}
```

The corresponding entry in `data_info.json` can be registered as:

```json
{
  "your_sft_dataset_name": {
    "file_name": "your_sft_dataset.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos",
      "audios": "audios"
    }
  }
}
```

---

#### DPO Data Format

The DPO data contains a user prompt, a chosen response, a rejected response, and the corresponding video/audio paths:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<video><audio>What visual is displayed as the song's instrumental fades out at the very end of the video?\nA. The artist's name 'ZAK DOWNTOWN'\nB. A black screen with the word 'MOODY'\nC. A globe logo with the text 'Downtown Worldwide'\nD. A red background with lightning\nPlease provide your answer by stating the letter followed by the full option, with a brief explanation grounded in the audio and visual cues."
    }
  ],
  "chosen": {
    "role": "assistant",
    "content": "In the video, From 2:44 onwards, the vocal track ends and the instrumental beat slowly fades out into silence; at the same moment in the visual, During this audio fade-out, a white globe logo with the text 'Downtown Worldwide' is displayed on a dark background. Therefore the correct answer is C. A globe logo with the text 'Downtown Worldwide'."
  },
  "rejected": {
    "role": "assistant",
    "content": "In the video, From 2:44 onwards, the vocal track ends and the instrumental beat slowly fades out into silence; at the same moment in the visual, During this audio fade-out, a white globe logo with the text 'Downtown Worldwide' is displayed on a dark background. Based on this, the answer is B. A black screen with the word 'MOODY'."
  },
  "videos": [
    "/path/to/video.mp4"
  ],
  "audios": [
    "/path/to/audio.wav"
  ]
}
```

The corresponding entry in `data_info.json` can be registered as:

```json
{
  "your_dpo_dataset_name": {
    "file_name": "your_dpo_dataset.json",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
      "messages": "messages",
      "chosen": "chosen",
      "rejected": "rejected",
      "videos": "videos",
      "audios": "audios"
    }
  }
}
```

Please modify the dataset names, file paths, and column mappings according to your local setup.

---

#### Training Stages

After registering the datasets, SFT and DPO can be launched using the standard LLaMA-Factory training interface. The exact command should be adjusted according to your hardware configuration, GPU memory, model size, and distributed training strategy.

Our training consists of two stages:

1. **Supervised Fine-Tuning (SFT)**  
   We first perform SFT to warm up the model on intervention-derived and audio-visual grounding data.

2. **Direct Preference Optimization (DPO)**  
   We then apply DPO using preference pairs that encourage audio-verified responses over visually plausible shortcut responses.

For the detailed hyperparameters used in our experiments, including learning rate, batch size, cutoff length, LoRA settings, DeepSpeed configuration, and training schedule, please refer to **Appendix C** in our paper.

---

### 🤗 Model Weights

The trained model checkpoint is available on Hugging Face:

**[wvs-thud-model](https://huggingface.co/Rakancorle1/wvs-thud-model)**

---

### 📁 Evaluation Data

The evaluation datasets and benchmark files used in THUD are currently being organized and will be released soon.

