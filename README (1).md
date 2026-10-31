# Generative AI - Practical No-05: Image Generation using Pretrained Stable Diffusion

## Text-to-Image Generation with Stable Diffusion v1.5 and Hugging Face Diffusers

A pretrained Stable Diffusion v1.5 model is loaded through the Hugging Face Diffusers library and used to generate images from text prompts. Three controlled experiments analyze how the **number of inference steps**, the **guidance scale** and the **sampling method (scheduler)** affect the quality of the generated images.

---

## 📋 Project Info

| Field                | Details           |
| -------------------- | ------------------ |
| **Student**          | Sneha Chaurasia    |
| **PRN**              | 202401110046       |
| **Class / Division** | CSE-AIML / A       |
| **Batch**            | A3                 |
| **Subject**          | Generative AI Lab  |
| **Practical No.**    | 05                 |

---

## 🎯 Problem Statement

Implement a pretrained Stable Diffusion model using the Hugging Face Diffusers library to generate images from text prompts. Analyze the impact of inference steps, guidance scale, and different sampling methods on the quality of the generated images.

## 🎯 Objectives

1. Load the pretrained Stable Diffusion v1.5 pipeline and generate an image from a text prompt.
2. **Experiment 1:** study the effect of inference steps (15, 30, 50).
3. **Experiment 2:** study the effect of guidance scale (3, 7.5, 12).
4. **Experiment 3:** compare the DDIM, Euler and DPM-Solver++ samplers.
5. Compare the outputs visually and by generation time.

---

## 📦 Model (no dataset needed)

This practical uses a **pretrained model for inference only**, so no dataset is required.

| Field          | Details                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------ |
| **Model**      | Stable Diffusion v1.5 (latent text-to-image diffusion model)                             |
| **Source**     | <https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5> (community mirror of the original RunwayML repo; fallback `sd-legacy/stable-diffusion-v1-5`) |
| **Library**    | Hugging Face Diffusers (`StableDiffusionPipeline`)                                       |
| **Output**     | 512 × 512 RGB images                                                                     |
| **Precision**  | fp16 on GPU                                                                              |
| **Loading**    | `StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)` — downloaded automatically (~4–5 GB) on the first run |

---

## 🧠 How Stable Diffusion Works

| Component              | Role                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| **Text encoder (CLIP)** | Converts the prompt into embeddings that condition the image generation                  |
| **U-Net**              | Denoising network; predicts the noise in the latent at every step, guided by the text    |
| **VAE**                | Decoder turns the final denoised 64×64×4 latent into a 512×512 image                      |
| **Scheduler**          | The sampling algorithm that decides how to step from noise to image                       |

Pipeline: `Prompt → CLIP embeddings → random latent noise → U-Net denoising (N steps, with classifier-free guidance) → VAE decoder → image`

---

## 🔬 Experiments

All experiments use the **same prompt and the same random seed (42)**, so only the parameter being studied changes.

**Prompt:** `a photo of an astronaut riding a horse on mars, highly detailed, cinematic lighting`
**Negative prompt:** `blurry, low quality, distorted, deformed`

| Experiment | Parameter varied | Values                      | Fixed settings                          |
| ---------- | ---------------- | --------------------------- | ---------------------------------------- |
| 1          | Inference steps  | 15, 30, 50                  | Guidance 7.5, default scheduler          |
| 2          | Guidance scale   | 3, 7.5, 12                  | 30 steps, default scheduler              |
| 3          | Sampler          | DDIM, Euler, DPM-Solver++   | Guidance 7.5, run at 20 and 10 steps     |

---

## 🔄 Workflow

1. Install and import libraries (`diffusers`, `transformers`, `accelerate`, PyTorch)
2. Check GPU availability
3. Load pretrained Stable Diffusion v1.5
4. Basic text-to-image generation
5. Experiment 1: inference steps
6. Experiment 2: guidance scale
7. Experiment 3: sampling methods
8. Display and compare all generated images, with a timing table
9. Observations, result and conclusion

---

## 📊 Results

> Replace the `XX.X` values with the numbers printed in the executed notebook (the timing table in the "Display and compare" cell).

**Generation time (seconds, T4 GPU):**

| Experiment | Setting                              | Time (s) |
| ---------- | ------------------------------------ | -------- |
| 1: Steps   | 15 steps                             | XX.X     |
| 1: Steps   | 30 steps                             | XX.X     |
| 1: Steps   | 50 steps                             | XX.X     |
| 2: Guidance | CFG 3                               | XX.X     |
| 2: Guidance | CFG 7.5                             | XX.X     |
| 2: Guidance | CFG 12                              | XX.X     |
| 3: Scheduler | DDIM, 20 steps                     | XX.X     |
| 3: Scheduler | Euler, 20 steps                    | XX.X     |
| 3: Scheduler | DPM-Solver++, 20 steps             | XX.X     |

**Graphs / images generated in the notebook:** the basic generated image, the three experiment grids, a combined 3×3 comparison grid, and a plot of generation time vs inference steps.

### Observations

> Check these against your own generated images and edit any line your results contradict.

- **Inference steps:** more steps give more refined detail, with diminishing returns after about 30 steps, while generation time grows roughly linearly with the step count.
- **Guidance scale:** a low scale (3) gives looser, more creative images; a high scale (12) follows the prompt strictly but tends toward over-saturation and artifacts; around 7.5 gives the best balance. Time is unaffected by the guidance scale.
- **Samplers:** all three cost about the same time per step, but differ in quality at low step counts, where the higher-order DPM-Solver++ is usually the most efficient.
- [Fill in after running: which sampler gave the best result at 10 steps in your run?]

---

## ✅ Conclusion

A pretrained Stable Diffusion v1.5 model was loaded with the Hugging Face Diffusers library and used to generate images from text prompts. The experiments show that image quality depends not only on the model but also on the sampling configuration: more steps improve detail with diminishing returns, the guidance scale trades creativity against prompt adherence, and an efficient solver such as DPM-Solver++ reaches good quality in fewer steps. A good practical setting is about **20–30 steps, a guidance scale of 7–8, and DPM-Solver++**.

**Limitations and future work:** the analysis is qualitative and based on one prompt and one seed. A stronger study would use several prompts and seeds with quantitative metrics (CLIP score, FID), and could be extended with negative-prompt studies, image-to-image generation, LoRA fine-tuning and newer models such as SDXL.

---

## 📁 Repository Structure

```
Image-Generation-using-Pretrained-Stable-Diffusion-Practical-No.-05-Gen-AI/
│
├── README.md                                       ← this file
├── Practical_05_Stable_Diffusion.ipynb            ← main notebook (executed, with outputs)
└── Sneha_Chaurasia_46_GenAI_Lab_StableDiffusion.pdf  ← assignment report
```

The model weights are downloaded from Hugging Face inside the notebook, so nothing else needs to be uploaded.

---

## ⚙️ How to Run

**Option 1 — Google Colab (recommended)**

1. Open `Practical_05_Stable_Diffusion.ipynb` in Google Colab (`File → Upload notebook`).
2. Select a GPU: `Runtime → Change runtime type → T4 GPU`.
3. Run all cells: `Runtime → Run all`. The first run downloads ~4–5 GB of weights.

**Option 2 — Local Jupyter (needs an NVIDIA GPU with ~6 GB+ VRAM)**

```
git clone https://github.com/Sneha529-oss/Image-Generation-using-Pretrained-Stable-Diffusion-Practical-No.-05-Gen-AI.git
cd Image-Generation-using-Pretrained-Stable-Diffusion-Practical-No.-05-Gen-AI
pip install diffusers transformers accelerate safetensors torch pandas matplotlib jupyter
jupyter notebook Practical_05_Stable_Diffusion.ipynb
```

The notebook takes roughly 5–10 minutes on a T4 GPU after the model is downloaded; on CPU it is impractically slow.

---

## 📦 Requirements

```
diffusers
transformers
accelerate
safetensors
torch
pandas
matplotlib
jupyter
```

---

## ✅ Submission Checklist

- [x] Code file (Jupyter Notebook, executed end-to-end)
- [x] Model source link included above
- [x] Pretrained Stable Diffusion v1.5 loaded with Diffusers
- [x] Experiments on inference steps, guidance scale and samplers
- [x] Comparison grid, timing table and observations
- [x] Result and conclusion
- [x] Report PDF
- [x] README file

---

## 📖 References

1. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. <https://arxiv.org/abs/2112.10752>
2. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. <https://arxiv.org/abs/2006.11239>
3. Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models (DDIM).* ICLR 2021. <https://arxiv.org/abs/2010.02502>
4. Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022. <https://arxiv.org/abs/2206.00364>
5. Lu, C. et al. (2022). *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models.* <https://arxiv.org/abs/2211.01095>
6. Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance.* <https://arxiv.org/abs/2207.12598>
7. Hugging Face Diffusers documentation: <https://huggingface.co/docs/diffusers>

---

## 📝 Declaration

I, **Sneha Chaurasia**, confirm that this implementation was prepared for Practical Assignment 5 and that the results presented are generated from the experiments performed in this notebook.

**GitHub Repository Link:** <https://github.com/Sneha529-oss/Image-Generation-using-Pretrained-Stable-Diffusion-Practical-No.-05-Gen-AI>
