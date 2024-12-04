import flash_attn
print("flash-attn imported successfully!")

from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
print("Flash Attention QKV Packed Function is available:", flash_attn_qkvpacked_func is not None)



