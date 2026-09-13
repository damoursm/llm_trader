@echo off
REM ============================================================================
REM  Production launcher for the LOCAL LLM server (Ollama), which serves the
REM  `local` sentiment engine (config/settings.py -> local_sentiment_base_url).
REM
REM  Point Task Scheduler at THIS file - same contract as run_scheduler.bat /
REM  run_dashboard.bat: a fixed environment, so the server is reproducible and
REM  does not depend on whatever shell happened to start it.
REM
REM  This is the STANDALONE (zip) Ollama, deliberately not the MSI installer:
REM  an installer runs Restart Manager, and on 2026-08-11 an installer's
REM  console-kill took down the whole trading stack for three hours
REM  (memory/restart-manager-kills-whole-stack-2026-08). A zip cannot do that.
REM
REM  OLLAMA_HOST      loopback only. The model server has no auth of its own;
REM                   binding wider would expose it to the LAN, and the
REM                   dashboard's own decommissioning made the same call.
REM  OLLAMA_MODELS    outside the repo - model blobs are GBs and must never be
REM                   a candidate for `git add`.
REM  OLLAMA_KEEP_ALIVE  the scheduler ticks every 30 min; -1 keeps the model
REM                   resident so a tick never pays the ~20s cold load. It is
REM                   ~5 GB of the 8 GB VRAM, and nothing else on this box uses
REM                   the GPU except an ad-hoc ml_ohlcv retrain (which runs on
REM                   CPU/RAM - see memory/torch-isolated-venv-2026-08).
REM  OLLAMA_CONTEXT_LENGTH  *** LOAD-BEARING ***. The default 4096, split across
REM                   parallel slots, gave an effective ~2048 tokens per request -
REM                   and Ollama truncates the OLDEST tokens SILENTLY, which is
REM                   exactly where the instruction prefix lives. Probed
REM                   2026-09-03 with a canary at the head of the prompt: at 2k
REM                   it was FOUND, at 5k and 8k it was LOST and the model
REM                   answered with filler words, with prompt_tokens pinned at
REM                   2050 and no error raised anywhere. The sentiment prompt is
REM                   ~2.4k tokens for 10 articles and ~4.7k at the 20-article
REM                   cap, so the default would have quietly degraded every large
REM                   digest. `options.num_ctx` in the request is NOT honoured by
REM                   the OpenAI-compatible endpoint - it must be set here.
REM  OLLAMA_KV_CACHE_TYPE  q8_0 (requires FLASH_ATTENTION) halves the KV cache so
REM                   8192 x 2 slots fits beside a 6.3 GB model in 8 GB of VRAM.
REM                   f16 at this context does NOT fit, and the failure mode of
REM                   not fitting is a silent CPU-offload slowdown, not an error.
REM  OLLAMA_NUM_PARALLEL  the sentiment pass fans out ~68 calls per tick; 2 slots
REM                   keep the GPU busy while leaving each request a full context.
REM ============================================================================

set OLLAMA_HOST=127.0.0.1:11434
set OLLAMA_MODELS=C:\Users\mathi\ollama\models
set OLLAMA_KEEP_ALIVE=-1
set OLLAMA_CONTEXT_LENGTH=8192
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
set OLLAMA_NUM_PARALLEL=2

"C:\Users\mathi\ollama\bin\ollama.exe" serve

REM If the server exits, %ERRORLEVEL% propagates so Task Scheduler can restart it.
exit /b %ERRORLEVEL%
