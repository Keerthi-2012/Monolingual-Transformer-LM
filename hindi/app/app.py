"""Local demo of the from-scratch Hindi 25M language model.

Tabs: (1) streaming text continuation with temperature / top-p / repetition-penalty controls,
(2) attention viewer (which earlier tokens each head looks at), (3) model facts and results.

Run (CPU is enough):
    python app.py --ckpt path/to/ckpt_best.pt            # then open http://127.0.0.1:7860
    python app.py --ckpt ... --selftest                  # generate one sample in the terminal and exit
"""
import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from transformer import GPT, GPTConfig  # noqa: E402
import sentencepiece as spm  # noqa: E402

TOKENIZER = ROOT / "tokenizer" / "models" / "hindi_bpe_24k.model"
EXAMPLES = ["भारत की राजधानी", "मुझे कल सुबह जल्दी उठकर", "विज्ञान और प्रौद्योगिकी के क्षेत्र में",
            "प्रधानमंत्री ने आज एक बैठक में कहा कि", "बच्चों को स्कूल में"]
DISCLAIMER = ("**Research demo.** A 25M-parameter model trained from scratch on 500M tokens of raw web and news Hindi. "
              "It writes locally fluent text but drifts, repeats and can state false or offensive things: the training "
              "data was not filtered for toxicity. Do not rely on its output.")

MODEL = SP = DEVICE = None


def load(ckpt_path):
    """Load model and tokenizer once. weights_only=True: the checkpoint only needs plain tensors and dicts."""
    global MODEL, SP, DEVICE
    DEVICE = "cpu"
    torch.set_num_threads(os.cpu_count() or 4)
    s = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    MODEL = GPT(GPTConfig(**s["config"]["model"])).to(DEVICE).eval()
    MODEL.load_state_dict(s["model"])
    SP = spm.SentencePieceProcessor(model_file=str(TOKENIZER))


@torch.inference_mode()
def stream_generate(prompt, max_new, temperature, top_p, rep_penalty):
    """Yield the growing text token by token. temperature 0 = greedy."""
    ids = SP.encode(prompt.strip()) or [SP.bos_id()]
    new = []
    eos = SP.eos_id()
    for _ in range(int(max_new)):
        x = torch.tensor([(ids + new)[-MODEL.cfg.block_size:]], device=DEVICE)
        logits = MODEL(x)[0][0, -1].float()
        if rep_penalty > 1.0:                                   # discourage tokens used recently (reduces loops)
            for t in set((ids + new)[-64:]):
                logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = F.softmax(logits / temperature, -1)
            sp_, si = probs.sort(descending=True)
            keep = sp_.cumsum(0) - sp_ <= top_p                 # nucleus: smallest set with mass >= top_p
            sp_ = sp_ * keep
            nxt = int(si[torch.multinomial(sp_ / sp_.sum(), 1)])
        if nxt == eos:
            break
        new.append(nxt)
        yield SP.decode(ids + new)


def generate_ui(prompt, max_new, temperature, top_p, rep_penalty):
    """Gradio generator: stream text and finish with a speed line."""
    t0, n, text = time.time(), 0, prompt
    for text in stream_generate(prompt, max_new, temperature, top_p, rep_penalty):
        n += 1
        yield text, f"{n} tokens, {n / (time.time() - t0):.1f} tokens/s"
    yield text, f"done: {n} tokens in {time.time() - t0:.1f}s ({n / max(time.time() - t0, 1e-9):.1f} tokens/s)"


def devanagari_font():
    for n in ("Nirmala UI", "Noto Sans Devanagari", "Mangal", "Lohit Devanagari"):
        try:
            font_manager.findfont(n, fallback_to_default=False)
            return n
        except ValueError:
            continue
    return None


@torch.inference_mode()
def attention_ui(text, layer, head):
    """Heatmap of one head: row = token being predicted from, column = earlier token it attends to."""
    ids = SP.encode(text.strip())[: MODEL.cfg.block_size]
    if len(ids) < 2:
        return None
    _, _, atts = MODEL(torch.tensor([ids]), return_attn=True)
    a = atts[int(layer)][0, int(head)].float().numpy()
    labels = [SP.id_to_piece(i).replace("▁", "") for i in ids]
    font = devanagari_font()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(a, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Layer {int(layer)}, head {int(head)}")
    ax.set_xlabel("Key position (token attended to)")
    ax.set_ylabel("Query position (token attending)")
    if font and len(ids) <= 40:
        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        ax.set_xticklabels(labels, rotation=90, fontname=font, fontsize=8)
        ax.set_yticklabels(labels, fontname=font, fontsize=8)
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    return fig


FACTS = """
### The model
- Decoder-only Transformer written from `nn.Linear` / `nn.Embedding` / `nn.LayerNorm` only (attention, causal mask and positional embeddings are hand-written)
- **25,383,552 parameters**: 9 layers, 6 heads, d_model 384, FFN 1536 (GELU), 512-token context, pre-norm, tied embeddings
- Custom SentencePiece BPE tokenizer (24k vocabulary), trained from scratch on Hindi
- Trained on **500M tokens** (OSCAR web text 78%, news 22%) for one epoch: about 4.3 hours on a single free T4 GPU

### Results (held-out text)
| | |
|---|---|
| Test perplexity / bits per byte | 37.75 / 0.513 |
| Generation BLEU-4 / chrF++ (T=0.5) | 3.1 / 13.0 (open-ended continuation, so low overlap is expected) |
| Greedy decoding | loops ("55% repeated 4-grams"), which the repetition penalty here reduces |

### What I found
- Attention is structured: middle layers (3-5) are the most focused and local; the first layer is nearly uniform
- Finetuning on a synthetic Hindi reasoning set reached **51%** vs **40%** random guessing in distribution, and **31%** on unseen phrasings (below chance on chain tasks): it learned the format and cues, not transferable reasoning
- Code, report and results: see the repository README
"""


def build_ui():
    import gradio as gr
    with gr.Blocks(title="Hindi 25M LM (from scratch)") as demo:
        gr.Markdown("# Hindi language model, built from scratch\n" + DISCLAIMER)
        with gr.Tab("Generate"):
            prompt = gr.Textbox(label="Hindi prompt", value=EXAMPLES[0], lines=3)
            gr.Examples(EXAMPLES, inputs=prompt)
            with gr.Row():
                max_new = gr.Slider(8, 200, value=60, step=4, label="Max new tokens")
                temp = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature (0 = greedy)")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                rep = gr.Slider(1.0, 1.6, value=1.15, step=0.05, label="Repetition penalty")
            btn = gr.Button("Generate", variant="primary")
            out = gr.Textbox(label="Model output", lines=8)
            speed = gr.Markdown()
            btn.click(generate_ui, [prompt, max_new, temp, top_p, rep], [out, speed])
        with gr.Tab("Attention viewer"):
            gr.Markdown("Type a short sentence and pick a layer and head. Bright cells show which earlier tokens a head "
                        "attends to when processing each token (upper triangle is empty: the model cannot see the future).")
            txt = gr.Textbox(label="Short Hindi sentence", value="भारत की राजधानी नई दिल्ली है और यह यमुना नदी के किनारे बसा है।")
            with gr.Row():
                layer = gr.Slider(0, 8, value=4, step=1, label="Layer (0-8)")
                head = gr.Slider(0, 5, value=0, step=1, label="Head (0-5)")
            show = gr.Button("Show attention")
            plot = gr.Plot()
            show.click(attention_ui, [txt, layer, head], plot)
        with gr.Tab("About the model"):
            gr.Markdown(FACTS)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("HINDI_LM_CKPT", str(Path(__file__).parent / "ckpt_best.pt")))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    load(a.ckpt)
    print(f"loaded {MODEL.n_params():,} parameters from {a.ckpt}", flush=True)
    if a.selftest:
        t0, n, text = time.time(), 0, ""
        for text in stream_generate(EXAMPLES[0], 40, 0.7, 0.9, 1.15):
            n += 1
        print(text)
        print(f"{n} tokens at {n / (time.time() - t0):.1f} tokens/s")
        sys.exit(0)
    build_ui().launch(server_name="127.0.0.1", server_port=a.port, inbrowser=True)
