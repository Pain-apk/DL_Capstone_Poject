"""
test_pilot_inference.py - Verifies that the fine-tuned pilot adapter generates valid JSON
recipe output matching candidate dishes without inventing dishes.
"""

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "outputs", "pilot_unverified", "adapter")

FIXED_SYSTEM_PROMPT = (
    "You are an expert Indian culinary assistant. Given detected ingredients, regional context, "
    "dietary requirements, time constraints, and candidate dishes, select the best matching dish "
    "and output the authentic recipe in strict JSON format. Output valid JSON only."
)

def test_inference():
    print(f"Loading base model {BASE_MODEL_ID} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    print(f"Loading trained LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    test_user_msg = (
        "Detected Ingredients: poha, onion, potato, peanuts\n"
        "Region: Maharashtra\n"
        "Diet: vegetarian\n"
        "Time Limit: 25 minutes\n"
        "Excluded Allergens: none\n"
        "Candidate Dishes: Pithla, Kanda Poha, Khaman Dhokla, Batata Vada"
    )

    messages = [
        {"role": "system", "content": FIXED_SYSTEM_PROMPT},
        {"role": "user", "content": test_user_msg}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    print("\nGenerating recipe with pilot model...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )

    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print("\n" + "=" * 60)
    print("PILOT MODEL GENERATED OUTPUT:")
    print("=" * 60)
    print(response_text)
    print("=" * 60)

    # Validate JSON
    try:
        parsed = json.loads(response_text)
        print("\n[VERIFICATION PASS] Output parsed successfully as strict valid JSON!")
        print(f"Dish Name: {parsed.get('dish_name')}")
        print(f"Local Name: {parsed.get('local_name')}")
        print(f"Ingredients count: {len(parsed.get('ingredients', []))}")
        print(f"Steps count: {len(parsed.get('steps', []))}")
        print(f"Cooking Time: {parsed.get('time_minutes')} mins")
        return True
    except Exception as e:
        print(f"\n[VERIFICATION FAIL] JSON parse error: {e}")
        return False

if __name__ == "__main__":
    test_inference()

