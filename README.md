# MoE-ERAS Submission Repository

## How do I replicate key results?

- Performance gains in generation: `notebooks/Speedup Profiling Submission.ipynb`
- Accuracy evaluation on Wikitext: ``
- Accuracy evaluation on C4: ``
- Activation pattern analysis: ``

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
9. Open up the respective notebook and GO! :D 

## Where is our implementation?

Repo `MoE_Expert_Scheduler` (`huggingface/transformers` fork)
 - Mixtral and Switch Transformer changes to collect router logits
 - Changes to generation utils to return required data

Repo `mixtral-offloading-residency-info` (`dvmazur/mixtral-offloading` fork)
 - Implementation of biasing and thresholding

