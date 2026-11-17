
import os, sys
hf_wrapper_base_path = os.environ.get('HF_WRAPPER_BASE_PATH')
if hf_wrapper_base_path is not None:
    sys.path.insert(0, hf_wrapper_base_path)

from model.rwkv7_backbone import RWKV7BackboneForCausalLM, RWKV7BackboneModel, MixerConfig
RWKV7BackboneForCausalLM = RWKV7BackboneForCausalLM
RWKV7BackboneModel = RWKV7BackboneModel
RWKV7BackboneConfig = MixerConfig
