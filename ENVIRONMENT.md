# Environment notes
* Pinned: `transformers==4.57.1`, `huggingface_hub==0.36.2` (assignment). All other versions: course environment lock (replace this note with its file if one is supplied).
* Every run writes `results/task1/environment.json` (python/torch/CUDA device/driver, package versions, git commit, checkpoint revision `config._commit_hash`, tokenizer fingerprint, attention backend, dtype, manifest SHA-256s, seed).
* Full runs need a CUDA GPU (T4, 16 GB is enough: FP16 model 3 GB + CPU snapshot; Fisher pass temporarily upcasts to FP32, ~6 GB). Unit tests and `--smoke` run on CPU with no network.
* Matmul TF32 is disabled globally (`set_seed`).
