# MoE-ERAS Submission Repository

## How do I replicate key results?

- Performance gains in generation: `notebooks/Speedup Profiling Submission.ipynb`
- Accuracy evaluation on Wikitext: `notebooks/wikitext_PPL_calculations.py`
- Accuracy evaluation on C4: `notebooks/C4_PPL_calculations.py`

## How do I run these notebooks?

All our code can be run on PACE.

1. Connect to gatech VPN.
2. Access `https://ondemand-ice.pace.gatech.edu/` and request and get a RHEL9 Interactive Desktop (from the top drop down) with H100.
3. Enter the VM's GUI and open up a terminal.
4. After setting up your git credentials, run `git clone git@github.com:abhibambhaniya/mixtral-offloading-residency-info.git`
5. `cd mixtral-offloading-residency-info`
6. `bash initial_setup.sh`
7. `conda activate $TMP_DIR/moe-offload`
8. `jupyter notebook`
9. `cd notebooks`
10. `huggingface-cli download lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo --quiet --cache-dir $TMP_DIR --local-dir Mixtral-8x7B-Instruct-v0.1-offloading-demo`
11. For performance gain results: Open up the speed up notebook and GO! :D
12. For quality results on wikitext/C4, run the respective python script. Ensure that step 10 completed successfully and verify that the local-dir matches the `state_path` in the script.  

## Where is our implementation?

Repo `MoE_Expert_Scheduler` (`huggingface/transformers` fork)
 - Mixtral and Switch Transformer changes to collect router logits
 - Changes to generation utils to return required data

Repo `mixtral-offloading-residency-info` (`dvmazur/mixtral-offloading` fork)
 - Implementation of biasing and thresholding

