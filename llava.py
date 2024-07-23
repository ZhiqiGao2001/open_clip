# Load model directly
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.5-7b")
model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")
model.eval()
