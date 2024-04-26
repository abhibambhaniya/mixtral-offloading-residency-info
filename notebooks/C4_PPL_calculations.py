

# In[2]:


# fix numpy in colab
import numpy

import os, sys
script_dir = os.getcwd()
module_path = script_dir
for _ in range(1):
    module_path = os.path.abspath(os.path.join(module_path, '../'))
    if module_path not in sys.path:
        sys.path.insert(0,module_path)
        
sys.path.append("mixtral-offloading")
import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from IPython.display import clear_output
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging
import time
import gc
from src.build_model import OffloadConfig, QuantConfig, build_model

import torch
from tqdm import tqdm
import pickle

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo2"

config = AutoConfig.from_pretrained(quantized_model_name)

device = torch.device("cuda:0")

##### Change this to 5 if you have only 12 GB of GPU VRAM #####
offload_per_layer = 4
# offload_per_layer = 5
###############################################################

num_experts = config.num_local_experts

offload_config = OffloadConfig(
    main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
    offload_size=config.num_hidden_layers * offload_per_layer,
    buffer_size=4,
    offload_per_layer=offload_per_layer,
)


attn_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=True,
)
attn_config["scale_quant_params"]["group_size"] = 256


ffn_config = BaseQuantizeConfig(
    nbits=2,
    group_size=16,
    quant_zero=True,
    quant_scale=True,
)
quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)


# del model

gc.collect
torch.cuda.empty_cache()


# In[ ]:


# In[7]:


from transformers import TextStreamer


tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
past_key_values = None
sequence = None

seq_len = 0
# while True:
user_input = "Where is Georgia Tech? What is the name of its mascot?"

user_entry = dict(role="user", content=user_input)
input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)

if past_key_values is None:
  attention_mask = torch.ones_like(input_ids)
else:
  seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
  attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)




# if 'model' in locals():
#     del model
# model = build_model(
#     device=device,
#     quant_config=quant_config,
#     offload_config=offload_config,
#     state_path=state_path,
#     routing_strategy="THRESHOLDING",
#     routing_threshold=0.25
# )





# ## Calculate Perplexity


# In[13]:


from datasets import load_dataset

test = load_dataset("allenai/c4", "en", split="validation", cache_dir='~/scratch/', streaming=True)
all_text = ""

for row in list(test.take(500)):
     all_text += "\n\n" + row['text']

encodings = tokenizer(all_text, return_tensors="pt")


# In[ ]:
for stride in [512, 1024]:
    for routing_config in [("TOP-K",0), ("THRESHOLDING",0.05) , ("THRESHOLDING",0.15), ("BIASING",0.25),  ("THRESHOLDING",0.1), ("THRESHOLDING",0.25)]:
    
        if 'model' in locals():
            del model
        model = build_model(
            device=device,
            quant_config=quant_config,
            offload_config=offload_config,
            state_path=state_path,
            routing_strategy=routing_config[0],
            routing_threshold=routing_config[1]
        )
    
        max_length = 4096
        # stride = 512
        seq_len = encodings.input_ids.size(1)
    
        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
    
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
    
                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss
    
            nlls.append(neg_log_likelihood)
    
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
            # print( torch.exp(torch.stack(nlls).mean()))
    
        ppl = torch.exp(torch.stack(nlls).mean())
    
    
    
    
    #     total_experts_saved = 0
    #     for i in result['router_logits'][-32:]:
    #         total_experts_saved += i[1]
    
        print(f"Config: {routing_config}, PPL:{ppl}")
    
    
        with open(f'nlls_C4_{routing_config[0]}_{routing_config[1]}_{max_length}_{stride}', 'wb') as fp:
            pickle.dump(nlls, fp)
    # In[ ]:


# print(ppl)


# In[ ]:




