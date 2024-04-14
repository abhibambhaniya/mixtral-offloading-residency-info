conda create -p $TMP_DIR/moe_offload python=3.10
conda activate $TMP_DIR/moe_offload
export PATH=$TMP_DIR/moe_offload/bin/:$PATH
pip install -r requirements.txt
