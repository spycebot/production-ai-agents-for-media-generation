# Exercise 1 Notes and Interactive Session

24 NOV 2026

## Jupyter Notebooks Fails to Run Out of the Box

**Query:**
<context>
I am running this Jupyter Notebook in Visual Studio Code. The cell execution appears to be hing at the cell 

```python
import sys
from pathlib import Path
```
</context>
<inputs>
<input index="1">
C:\Users\safetystation\Documents\production-ai-agents-for-media-generation\01-consistent-media-generation-on-serverless-gpus\code\nb-first-generation-local.ipynb
</input>
<input index="2">
https://github.com/DataTalksClub/production-ai-agents-for-media-generation/blob/main/01-consistent-media-generation-on-serverless-gpus/02-first-generation.md
</input>
</inputs>
<query>
Please help me diagnose the issue with this notebook execution.
</query>

**Response:**
The `import sys` cell isn't hanging. It's the first cell you ran, so it's waiting for a Jupyter kernel that never starts. Any cell would stop at the same spot.

**Why the kernel can't start**

| Check | What I found |
|---|---|
| Python on this machine | Only **3.14.4** |
| Python the repo supports | `requires-python = ">=3.12,<3.14"` in `pyproject.toml`, so 3.14 is outside it |
| `ipykernel` in that Python | **Not installed** (`ModuleNotFoundError`) |
| Project virtual env (`.venv`) | **Doesn't exist**. The repo uses `uv` (`uv.lock`), but `uv` isn't installed |
| Registered Jupyter kernels | None |

The notebook's kernel setting is just a generic "Python 3". VS Code points that at your system Python 3.14, which has no `ipykernel`, so the cell sits on its spinner. The output panel usually shows a kernel-start error or an "install ipykernel?" prompt that's easy to miss.

The saved outputs in the notebook came from the course author's Mac (paths like `/Users/mnrozhkov/...`). They don't show that it ever ran on your machine.

**Fix**, from the repo root (`production-ai-agents-for-media-generation`):

1. Install uv:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
2. Open a new terminal, then create the environment. uv downloads a compatible Python 3.13 on its own:
```powershell
uv sync --extra notebooks
```
3. In VS Code, click the kernel picker at the top right of the notebook. Choose **Select Another Kernel → Python Environments** and pick `.venv\Scripts\python.exe` from the repo.
4. Restart the kernel and run the cells from the top.

You don't need the `gpu` extra on your laptop. This notebook only calls the remote Endpoint.

One more thing: the next cell (`setup_endpoint()`) asks for the URL and token with an input box at the top of VS Code. If execution seems stuck there, look for that prompt.

-----

### HTTPError

Query:
The output for the previous cell now says "Endpoint configured (***kew2) ". However, when I run the next cell

```python
models = wait_ready(BASE_URL, TOKEN, deadline_s=600)
print("Served models:", [m.get("id") for m in models["data"]])
```

it gives and `HTTPError` .

**Response:**
It's an **HTTP 404**, and it comes from Nebius's tunnel, not from the model server. I sent a request to your Endpoint URL without the token, and Nebius's proxy answered:

```
HTTP/1.1 404 Not Found
Proxy-Status: nebius-tunnel; error=proxy_internal_response; details="unknown_tunnel"
```

A made-up hostname on the same domain gets exactly the same reply. So Nebius doesn't currently know a tunnel at `https://port8000-jxcws3qypmkk60a.tunnel.applications.eu-west2.nebius.cloud`. The request never reaches your model, so the token isn't the issue yet.

`wait_ready` only retries 429/502/503/504, the "still starting up" errors. A 404 is raised straight away, which is why the cell stopped with `HTTPError`.

**Likely causes, most likely first:**
1. **The Endpoint isn't `RUNNING`.** It may be stopped, preempted, still provisioning, or deleted. Check its status in the Nebius console.
2. **The URL is from a different Endpoint**, for example an earlier one you deleted and recreated. Each Endpoint gets a new tunnel hostname.
3. **A typo** in the pasted URL. Compare it character by character with the one in the console.

**To fix it:**
- Once the console shows `RUNNING`, copy its HTTPS URL. If it matches the one above, just re-run the `wait_ready` cell.
- If the URL has changed, open `code/.env` and delete the `BASE_URL` line. If it's a new Endpoint, delete the `TOKEN` line too. Then restart the kernel and run the cells again from the top. Step 1 will ask for the new values.

After the URL is right, you may see `502` for a few minutes while the model loads. The cell waits through that by itself.

-----

### Improving How the Eyes are Rendered

**Query:**
It is interesting to use a cloud GPU to generate media assets. However, in the Practice** section, I am unhappy with the depiction of the girl, because her eyes do not align (we would say that she is cross-eyed (eyes pointing in) or wall-eyed (eyes pointing out)). If it were my job to produce these images, how would I correct this issue with the eyes so that the images were more appealing?

**Response:**
Congratulations on getting the homework submitted. I looked at your Practice cells, and a few things in the setup make wandering eyes likely. Here's how I'd fix them, cheapest first.

**Why it happens here**
- **The face is tiny.** You ask for `512x512` and a full figure: girl, dress, hat, cake, plus snow, a laptop or decorations. The face may only be about 50 pixels across, so each eye is a handful of pixels. Diffusion models handle small details badly, and eyes are where people notice it most. The model you're using, `Sana_1600M_1024px`, was also trained for **1024×1024**, so 512 is below its native size.
- **The style line asks for loose, hand-drawn faces.** "Ink and pigment illustration in the style of …" pulls toward sketchy picture-book drawing, where eyes are often just a pair of quick marks. Naming a specific published book also runs against the course's rule to keep prompts free of known works. Describing the style's traits works better anyway.

**Fixes at generation time** (all available in `generate_image` / `ImageGenerator`)
1. **Generate at 1024×1024** (`size="1024x1024"`). This is the biggest single improvement for faces.
2. **Frame the face larger.** Ask for a "medium shot, waist up, face clearly visible, looking at the viewer" so the face takes up more of the image.
3. **Describe the eyes outright:** e.g. "symmetrical friendly face, both eyes looking at the viewer". In a picture-book style, "simple round dot eyes" avoids the problem almost entirely.
4. **Add a negative prompt.** `ImageGenerator.generate_image` accepts `negative_prompt=` (it's `""` by default in `models.py`). Try "cross-eyed, misaligned eyes, asymmetrical eyes, deformed face". It helps, but it won't guarantee a fix.
5. **Slightly more steps.** The default is `num_inference_steps=20`; 28–30 often sharpens fine detail. You can leave `guidance_scale` at 4.5.
6. **Generate several seeds and pick the best.** Each image takes about a second on the Endpoint, so for a real job you'd make 8–16 variants per scene and keep the good ones. That's normal practice, and you record the winning seed so you can reproduce it.

**Fixing an image you otherwise like** (the professional route)
- **Edit just the face.** Use an image-editing model with an instruction like "correct her eyes so both look at the viewer; keep everything else unchanged". This course's own landscape lesson lists **Qwen-Image-Edit-2511** for this, and it has a Nebius template, so you'll meet it later in the module.
- **Inpaint the eyes:** mask only the eye area and regenerate it, leaving the rest of the image untouched.
- **Run a "face detailer" pass:** crop the face, regenerate that crop at high resolution, then paste it back. This is the standard fix for small faces in full-body shots.

For your birthday-card images, I'd try 1024×1024, waist-up framing, "simple round dot eyes looking at the viewer" in the prompt, and a handful of seeds. That will probably fix it without any editing step.

-----

### Projected Cost Per Asset at N = 1000, in USD

**Query:**
https://courses.datatalks.club/production-ai-agents-generative-media/homework/module1 Question 7 in he homework asks "What is your projected cost per asset at N = 1000, in USD?" How do I determine this. Was this something I should have read from the Nebius cloud endpoint before I deleted it?

**Response:**
You don't need anything more from the deleted Endpoint. At N = 1000, the times you could only have read there (startup and model load) change the answer by about a tenth of a cent per thousand assets.

**How the course computes it** (lesson 05 and the homework instructions):

> **Cost per asset = F / N + c**
> - **F** (fixed, paid once per run) = (startup + model load seconds) × price per second
> - **c** (paid per asset) = median generation seconds × price per second

**Your numbers**

- **Q6, median generation time: 1.9 s.** I pulled all 27 `Generated in … s` lines from your saved notebook. They range from 1.7 to 2.0 s, and the median is **1.9**. These times are measured from your laptop, so they include network time and slightly overstate GPU time. The homework says any honest answer counts, so that's fine.
- **Price:** Nebius doesn't charge separately for Endpoints; they're billed at Compute VM prices. Nebius's pricing docs list the L40S, available only in eu-north1 (your Endpoint's region), at **$0.65 per GPU-hour preemptible** and **$1.35 on-demand**. The Sana template is preemptible by default.
- **F:** endpoint mode doesn't record startup or load (`startup_s: null` in your `metrics.json`). Your earlier B300 Endpoint's logs show about 4–4.5 minutes from "Provisioning" to the model loaded, so use **about 260 s** as an estimate.

**Q7 at N = 1000, preemptible ($0.65/h = $0.000181/s):**
- c = 1.9 × 0.000181 ≈ $0.000343
- F/N = 260 × 0.000181 / 1000 ≈ $0.000047
- **Total ≈ $0.0004 per asset**

If your Endpoint was **not** preemptible ($1.35/h), it's about **$0.0008**.

Your fixed cost could be off by a minute either way and the answer would still round to $0.0004. That's the lesson's point: with large batches, cost per asset is almost entirely generation time × price.

If you want to confirm which rate you actually paid, **Billing → Usage** in the Nebius console still shows the deleted Endpoint's charges. The line item should say whether it was billed as preemptible.

-----

### Contributions to FAQ

**Query:**
For the previous two courses, I did not make any FAQ contribution along with the homework. I would like to begin doing so now. How do I go about this? Do I make a modification to the repo?

**Response:**
No, you don't change the course repo. DataTalksClub keeps all its course FAQs in a separate repo, **DataTalksClub/faq**, and new entries are proposed through a **GitHub issue form**, not a pull request.

**How to contribute** (from that repo's `CONTRIBUTING.md`):
1. Check the course FAQ first, so you don't duplicate an existing entry: https://datatalks.club/faq/production-ai-agents-generative-media.html
2. Open the proposal form: https://github.com/DataTalksClub/faq/issues/new?template=faq-proposal.yml
3. Pick the course (**Production AI Agents for Generative Media**), then write the question and your answer.
4. A maintainer, or their automation, turns it into an entry and reviews it before it's published. You'll get a comment on the issue either way, including if it's already answered.

Pull requests are only for fixing an **existing** entry.

**Their writing tips, in short:**
- Phrase the question the way a student would search for it, starting with How, What, Why or When.
- Cover one problem per entry.
- Give the fix first.
- Make any code runnable as written.
- Link to official sources instead of paraphrasing them.
- If unclear course material caused the confusion, say so.

**What to put in the homework form:** neither the FAQ page nor the contribution guide says. The link to your proposal issue (`https://github.com/DataTalksClub/faq/issues/…`) is the natural thing to submit. If the form asks for something else, the course Slack channel `#course-ai-mediagen` can confirm.

**You already have good material from today**, each a real problem another student could hit:
- **"Why does my Sana Endpoint go to ERROR with 'no kernel image is available for execution on the device'?"** The Endpoint landed on a B300, and the container's PyTorch (cu126) only supports up to sm_90. Pick an L40S (or H100/H200) in the form. This is probably the most valuable one.
- **"Why does `wait_ready` fail with HTTPError 404?"** `unknown_tunnel` means the Endpoint isn't running (stopped, crashed or deleted), or the URL is out of date. Check its status and logs in the console.
- **"Is the `tenantuseraccount-…` string in the console my Endpoint token?"** No, it's your account ID. The token is shown once in the create form.
- **"Why does the first notebook cell hang in VS Code on Windows?"** No kernel starts with system Python 3.14. Run `uv sync --extra notebooks` and select `.venv`.
