"""
src/infer.py - Interactive & CLI inference runner for the fine-tuned Recipe LLM.
Supports pre-configured regional scenarios, custom arguments, or interactive mode.
"""

import os
import sys
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "outputs", "pilot_unverified", "adapter")

FIXED_SYSTEM_PROMPT = (
    "You are an expert Indian culinary assistant. Given detected ingredients, regional context, "
    "dietary requirements, time constraints, and candidate dishes, select the best matching dish "
    "and output the authentic recipe in strict JSON format. Output valid JSON only."
)

SAMPLE_SCENARIOS = [
    {
        "name": "Scenario 1: Maharashtrian Breakfast (Veg)",
        "ingredients": "poha, onion, green chilli, mustard seeds, curry leaves, peanuts",
        "region": "Maharashtra",
        "diet": "vegetarian",
        "time": "25 minutes",
        "allergens": "none",
        "candidates": ["Kanda Poha", "Upma", "Batata Vada", "Thalipeeth"]
    },
    {
        "name": "Scenario 2: Spicy Kolhapuri Non-Veg Curry (Chicken)",
        "ingredients": "chicken, onion, dry coconut, red chilli, ginger garlic paste, garam masala",
        "region": "Maharashtra",
        "diet": "non-vegetarian",
        "time": "45 minutes",
        "allergens": "none",
        "candidates": ["Kolhapuri Sukka Chicken", "Tambda Rassa", "Mutton Sukka", "Chicken Maratha"]
    },
    {
        "name": "Scenario 3: Classic Stuffed Brinjal Curry (Veg)",
        "ingredients": "small brinjal, roasted peanuts, sesame seeds, coconut, goda masala, tamarind, jaggery",
        "region": "Maharashtra",
        "diet": "vegetarian",
        "time": "35 minutes",
        "allergens": "none",
        "candidates": ["Bharli Vangi", "Pithla", "Shev Bhaji", "Batata Bhaji"]
    },
    {
        "name": "Scenario 4: Traditional Comfort Legume Curry (Sprouted Matki)",
        "ingredients": "sprouted matki, onion, tomato, goda masala, mustard seeds, curry leaves",
        "region": "Maharashtra",
        "diet": "vegetarian",
        "time": "30 minutes",
        "allergens": "none",
        "candidates": ["Matki Usal", "Misal Pav", "Varan", "Katachi Amti"]
    }
]

def load_pilot_model():
    print(f"[*] Loading base model '{BASE_MODEL_ID}' (4-bit NF4 quantization)...")
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
    print(f"[*] Loading LoRA adapter from '{ADAPTER_PATH}'...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("[*] Model loaded and ready for inference.\n")
    return model, tokenizer

def run_single_inference(model, tokenizer, ingredients, region, diet, time_limit, allergens, candidates):
    user_prompt = (
        f"Detected Ingredients: {ingredients}\n"
        f"Region: {region}\n"
        f"Diet: {diet}\n"
        f"Time Limit: {time_limit}\n"
        f"Excluded Allergens: {allergens}\n"
        f"Candidate Dishes: {candidates}"
    )

    messages = [
        {"role": "system", "content": FIXED_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=450,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    raw_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return raw_response

def format_recipe_display(raw_json_str):
    try:
        data = json.loads(raw_json_str)
        dish = data.get("dish_name", "Unknown")
        local_name = data.get("local_name", "")
        time_mins = data.get("time_minutes", "N/A")
        ingredients = data.get("ingredients", [])
        steps = data.get("steps", [])
        missing = data.get("missing_ingredients", [])

        out = []
        out.append(f"Dish: {dish}" + (f" ({local_name})" if local_name and local_name != dish else ""))
        out.append(f"Estimated Time: {time_mins} minutes")
        out.append("\nIngredients Required:")
        for ing in ingredients:
            item = ing.get("item", "")
            qty = ing.get("quantity", "")
            out.append(f"  - {item}: {qty}")
        if missing:
            out.append(f"\nPantry Gaps (Pantry lacks): {', '.join(missing)}")
        out.append("\nCooking Steps:")
        for idx, step in enumerate(steps, 1):
            out.append(f"  {idx}. {step}")
        return "\n".join(out), True
    except Exception as e:
        return f"[Raw Output]: {raw_json_str}\n(JSON Parse Warning: {e})", False

def auto_select_candidates(ingredients_str, region, diet):
    """
    Auto-select 4 candidate dishes matching region and diet if not explicitly provided.
    """
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra_sources.csv")
    if not os.path.exists(csv_path):
        return ["Kanda Poha", "Pithla", "Batata Vada", "Misal Pav"]
    
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        # Filter by diet if specified
        diet_clean = diet.strip().lower()
        if "non" in diet_clean:
            df_filtered = df[df['diet'].str.lower().str.contains("non", na=False)]
        elif "veg" in diet_clean:
            df_filtered = df[df['diet'].str.lower() == "vegetarian"]
        else:
            df_filtered = df
        
        # Filter by region if specified
        if region:
            df_reg = df_filtered[df_filtered['state'].str.lower() == region.strip().lower()]
            if len(df_reg) >= 4:
                df_filtered = df_reg

        # Look for keyword overlap with dish names
        ing_words = [w.strip().lower() for w in ingredients_str.replace(",", " ").split() if len(w) > 3]
        matches = []
        for _, row in df_filtered.iterrows():
            d_name = str(row['dish_name']).lower()
            score = sum(1 for w in ing_words if w in d_name)
            matches.append((score, row['dish_name']))
        
        matches.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [m[1] for m in matches[:4]]
        if len(top_candidates) < 4:
            top_candidates += [d for d in df_filtered['dish_name'].head(4) if d not in top_candidates]
        return top_candidates[:4]
    except Exception:
        return ["Kanda Poha", "Pithla", "Batata Vada", "Misal Pav"]

def main():
    parser = argparse.ArgumentParser(description="Recipe LLM Inference Runner")
    parser.add_argument("--ingredients", type=str, help="Comma-separated detected ingredients")
    parser.add_argument("--region", type=str, default="Maharashtra", help="Culinary region")
    parser.add_argument("--diet", type=str, default="vegetarian", help="Dietary constraint (vegetarian/non-vegetarian/vegan)")
    parser.add_argument("--time", type=str, default="30 minutes", help="Time limit constraint")
    parser.add_argument("--allergens", type=str, default="none", help="Excluded allergens")
    parser.add_argument("--candidates", type=str, help="Comma-separated candidate dishes")
    parser.add_argument("--all_samples", action="store_true", help="Run through all pre-defined test scenarios")
    args = parser.parse_args()

    model, tokenizer = load_pilot_model()

    if args.ingredients:
        if args.candidates:
            candidates_list = [c.strip() for c in args.candidates.split(",") if c.strip()]
        else:
            candidates_list = auto_select_candidates(args.ingredients, args.region, args.diet)
            print(f"[*] Auto-selected candidate dishes matching pantry & regional profile: {', '.join(candidates_list)}")

        cands_str = ", ".join(candidates_list)
        print("=" * 70)
        print("RUNNING CUSTOM QUERY")
        print("=" * 70)
        print(f"Pantry: {args.ingredients}")
        print(f"Region: {args.region} | Diet: {args.diet} | Time Limit: {args.time}")
        print(f"Candidates: {cands_str}")
        print("-" * 50)
        raw_res = run_single_inference(
            model, tokenizer,
            args.ingredients, args.region, args.diet, args.time, args.allergens, cands_str
        )
        formatted, is_valid = format_recipe_display(raw_res)
        print("\n" + formatted)
        print("\n[Valid JSON]:", is_valid)
        print("=" * 70)
    else:
        print("=" * 70)
        print("RUNNING DIVERSE DEMONSTRATION SCENARIOS")
        print("=" * 70)
        for sc in SAMPLE_SCENARIOS:
            print(f"\n>>> {sc['name']}")
            print(f"Pantry: {sc['ingredients']}")
            print(f"Candidates: {', '.join(sc['candidates'])}")
            print("-" * 50)
            cands_str = ", ".join(sc["candidates"])
            raw = run_single_inference(
                model, tokenizer,
                sc["ingredients"], sc["region"], sc["diet"], sc["time"], sc["allergens"], cands_str
            )
            formatted, is_valid = format_recipe_display(raw)
            print(formatted)
            print(f"\n[Schema Check: {'PASSED (Valid JSON)' if is_valid else 'FAILED'}]")
            print("=" * 70)

if __name__ == "__main__":
    main()

