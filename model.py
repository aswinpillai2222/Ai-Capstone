from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login
import torch

# Initialize model and tokenizer only once
if "llm" not in globals():
    login(token="...")
    model_name = "mistralai/Mistral-7B-v0.1"

    # Optimized quantization config for accuracy and efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                  # Keep 4-bit for memory efficiency
        bnb_4bit_use_double_quant=True,     # Double quantization improves precision
        bnb_4bit_quant_type="nf4",          # NF4 is optimized for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 balances speed and accuracy
        llm_int8_skip_modules=["lm_head"],  # Skip quantizing lm_head for better output quality
        llm_int8_enable_fp32_cpu_offload=False  # No CPU offload, keep it on GPU
    )

    # Clear GPU memory before loading
    torch.cuda.empty_cache()

    # Load tokenizer with padding settings for consistency
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Ensure padding is set

    # Load model with optimized settings
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",              # Automatic layer mapping to GPU
        low_cpu_mem_usage=True,         # Minimize CPU memory usage
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,     # Consistent bfloat16 precision
        max_memory={0: "7.5GB"},        # Cap VRAM at 7.5GB for safety
    )

def ask_llm(prompt):
    try:
        # Tokenize with moderate max_length for better context retention
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")  # Increased to 384 for accuracy
        
        # Generate with tuned parameters for better accuracy
        with torch.no_grad():  # Save memory by disabling gradients
            outputs = llm.generate(
                **inputs,
                max_new_tokens=256,         # Limit response length for conciseness
                temperature=0.5,            # Lowered for less randomness, more precision
                top_p=0.7,                  # Tightened for focused sampling
                top_k=20,                   # Added top-k sampling for better token selection
                do_sample=True,             # Sampling for varied but controlled output
                repetition_penalty=1.2,     # Penalize repetition for coherent responses
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()
    except Exception as e:
        print("Error during generation:", e)
        return "Sorry, I encountered an error while processing your request."

def summarize_response(response):
  summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
  summary = summarizer(response, max_length=100, min_length=30, do_sample=False)
  cleaned_response = summary[0]['summary_text']
  return cleaned_response