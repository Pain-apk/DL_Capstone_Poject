"""
update_pilot_and_sources.py - Updates maharashtra_sources.csv with new verification schema,
and writes data/raw/maharashtra_drafts.csv for the 10 pilot dishes using verified search data.
"""

import os
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCES_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra_sources.csv")
DRAFTS_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra_drafts.csv")

# 10 Pilot dishes with verified real web sources (opened on 2026-11-07), own-word directions, and exact metadata
PILOT_DISHES = [
    {
        "dish_name": "Kanda Poha",
        "romanized_local_name": "Kande Pohe",
        "native_name": "कांदे पोहे",
        "native_name_checked": "yes",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_minutes": 20,
        "servings": 2,
        "main_ingredients": "poha|onion|peanuts|green chilli",
        "pantry_items": "oil|mustard seeds|cumin seeds|turmeric|curry leaves|salt|lemon juice|coriander leaves|sugar",
        "ingredients_with_quantity": "1.5 cup poha|1 medium onion|3 tbsp roasted peanuts|2 green chilli|2 tbsp oil|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1/2 tsp turmeric|8 curry leaves|1/2 tsp sugar|1 tbsp lemon juice|2 tbsp coriander leaves|salt to taste",
        "steps": "Rinse poha in a colander until softened and drain thoroughly without soaking|Roast or fry raw peanuts in 1 tbsp oil until golden crisp then set aside|In remaining oil splutter mustard seeds cumin seeds and saute green chillies curry leaves and chopped onions until soft|Stir in turmeric powder salt and sugar then add rinsed poha and fried peanuts|Gently toss on low flame cover and steam for 2 minutes|Turn off flame drizzle fresh lemon juice and garnish with chopped coriander leaves",
        "source_1": "https://hebbarskitchen.com/kanda-poha-recipe-traditional-maharashtrian/",
        "source_2": "https://www.vegrecipesofindia.com/kanda-poha-recipe-how-to-make-kanda-poha/",
        "source_3": "https://www.tarladalal.com/kanda-poha-3765r",
        "url_checked": "yes",
        "verified": "no",
        "review_flag": "pending_pilot_approval"
    },
    {
        "dish_name": "Misal Pav",
        "romanized_local_name": "Misal Pav",
        "native_name": "मिसळ पाव",
        "native_name_checked": "yes",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_minutes": 35,
        "servings": 3,
        "main_ingredients": "matki|onion|tomato|farsan",
        "pantry_items": "oil|mustard seeds|cumin seeds|ginger|garlic|turmeric|red chilli powder|goda masala|salt|coriander leaves|lemon",
        "ingredients_with_quantity": "2 cup matki|2 medium onion|2 medium tomato|1 cup farsan|3 tbsp oil|1 tsp mustard seeds|1 tsp cumin seeds|1 tbsp ginger garlic paste|1/2 tsp turmeric|1.5 tsp red chilli powder|2 tsp goda masala|salt to taste|2 tbsp coriander leaves|1 lemon",
        "steps": "Boil sprouted matki with turmeric salt and water until tender but holding shape|Heat oil in a heavy pot splutter mustard cumin seeds and saute onions until golden|Add ginger garlic paste and tomatoes cooking until oil separates at edges|Stir in red chilli powder goda masala and boiled matki with its cooking liquor|Pour 2.5 cups water simmer on low flame for 12 minutes until red oil tarri rises to surface|Assemble by ladling hot sprouts and spicy rassa into bowls topping with farsan raw onions and fresh lemon juice",
        "source_1": "https://hebbarskitchen.com/kolhapuri-misal-pav-recipe/",
        "source_2": "https://www.vegrecipesofindia.com/misal-pav-recipe-puneri-misal-pav/",
        "source_3": "https://www.archanaskitchen.com/maharashtrian-kolhapuri-misal-pav-recipe",
        "url_checked": "yes",
        "verified": "no",
        "review_flag": "pending_pilot_approval"
    },
    {
        "dish_name": "Thalipeeth",
        "romanized_local_name": "Thalipeeth",
        "native_name": "थालीपीठ",
        "native_name_checked": "yes",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_minutes": 25,
        "servings": 2,
        "main_ingredients": "jowar|besan|atta|onion",
        "pantry_items": "oil|cumin seeds|turmeric|coriander powder|green chilli|coriander leaves|salt|sesame seeds",
        "ingredients_with_quantity": "1 cup jowar|1/4 cup besan|1/4 cup atta|1 medium onion|2 green chilli|1 tbsp sesame seeds|1 tsp cumin seeds|1/2 tsp turmeric|1 tsp coriander powder|2 tbsp coriander leaves|2 tbsp oil|salt to taste",
        "steps": "Combine jowar besan and atta in a mixing bowl with finely chopped onions green chillies and fresh coriander|Add sesame seeds cumin powder turmeric and salt then gradually knead with warm water into soft pliable dough|Divide dough into lemon sized balls and grease a damp parchment sheet or banana leaf with oil drops|Pat dough ball flat using wet fingertips into a circular disc poking three small steam holes in center|Transfer patted thalipeeth onto medium hot griddle drizzle oil into holes and around edges|Roast covered for 2 minutes flip and toast uncovered until crisp golden brown spots form on both sides",
        "source_1": "https://www.vegrecipesofindia.com/thalipeeth-recipe/",
        "source_2": "https://hebbarskitchen.com/instant-thalipeeth-recipe-bhajani/",
        "source_3": "https://www.tarladalal.com/thalipeeth-2708r",
        "url_checked": "yes",
        "verified": "no",
        "review_flag": "pending_pilot_approval"
    },
    {
        "dish_name": "Pithla",
        "romanized_local_name": "Pithla",
        "native_name": "पिठलं",
        "native_name_checked": "yes",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "no",
        "time_minutes": 15,
        "servings": 2,
        "main_ingredients": "besan|onion|garlic|green chilli",
        "pantry_items": "oil|mustard seeds|cumin seeds|hing|curry leaves|turmeric|salt|water|coriander leaves",
        "ingredients_with_quantity": "1 cup besan|1 large onion|6 clove garlic|2 green chilli|2 tbsp oil|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1 pinch hing|8 curry leaves|1/2 tsp turmeric|salt to taste|2.5 cup water|2 tbsp coriander leaves",
        "steps": "Whisk besan with 1.5 cups water in a bowl until completely smooth and lump free|Heat oil in a kadhai crackle mustard seeds cumin seeds and saute crushed garlic green chillies curry leaves and hing|Add chopped onions and saute on medium flame until translucent and fragrant|Pour in prepared besan slurry followed by remaining cup of water turmeric and salt while stirring constantly|Cover pot and cook on low heat for 6 minutes stirring at intervals until thickened to a custard consistency|Garnish with fresh coriander leaves and serve piping hot alongside jowar bhakri and garlic thecha",
        "source_1": "https://www.vegrecipesofindia.com/pitla-recipe-maharashtrian-pitla/",
        "source_2": "https://hebbarskitchen.com/pithla-recipe-pitla-bhakri/",
        "source_3": "https://www.archanaskitchen.com/maharashtrian-pithla-recipe",
        "url_checked": "yes",
        "verified": "no",
        "review_flag": "pending_pilot_approval"
    },
    {
        "dish_name": "Matki Usal",
        "romanized_local_name": "Matki Usal",
        "native_name": "मटकी उसळ",
        "native_name_checked": "yes",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_minutes": 20,
        "servings": 2,
        "main_ingredients": "matki|onion|garlic|coconut",
        "pantry_items": "oil|mustard seeds|cumin seeds|hing|curry leaves|turmeric|red chilli powder|goda masala|jaggery|salt|coriander leaves",
        "ingredients_with_quantity": "2 cup matki|1 medium onion|4 clove garlic|2 tbsp coconut|2 tbsp oil|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1 pinch hing|8 curry leaves|1/2 tsp turmeric|1 tsp red chilli powder|1 tsp goda masala|1/2 tsp jaggery|salt to taste|1 cup water|2 tbsp coriander leaves",
        "steps": "Heat oil in a pan crackle mustard cumin seeds and sizzle hing with curry leaves|Add chopped onion and minced garlic sauteing until light golden brown|Stir in turmeric red chilli powder and authentic goda masala on low heat for 30 seconds|Toss in fresh sprouted matki grated coconut jaggery and salt mixing well|Add 1 cup water cover with lid and simmer on medium low heat for 8 to 10 minutes until sprouts soften|Turn off heat garnish with fresh coriander leaves and serve with hot chapatis or bhakri",
        "source_1": "https://www.vegrecipesofindia.com/matki-usal-recipe/",
        "source_2": "https://hebbarskitchen.com/matki-chi-usal-recipe-matki-ki-sabji/",
        "source_3": "https://www.tarladalal.com/matki-chi-usal-3870r",
        "url_checked": "yes",
        "verified": "no",
        "review_flag": "pending_pilot_approval"
    },
    {
        "dish_name": "Bharli Vangi",
        "romanized_local_name": "Bharli Vangi",
        "native_name": "भरली वांगी",
        "native_name_checked": "yes",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_minutes": 30,
        "servings": 3,
        "main_ingredients": "eggplant|peanuts|coconut|onion",
        "pantry_items": "oil|mustard seeds|cumin seeds|hing|curry leaves|ginger|garlic|turmeric|goda masala|red chilli powder|salt|jaggery|coriander leaves",
        "ingredients_with_quantity": "8 small eggplant|1/4 cup roasted peanuts|3 tbsp dry coconut|1 medium onion|1 tbsp ginger garlic paste|2.5 tbsp oil|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1 pinch hing|8 curry leaves|1/2 tsp turmeric|1.5 tsp goda masala|1 tsp red chilli powder|1 tsp jaggery|salt to taste|1 cup water|2 tbsp coriander leaves",
        "steps": "Slit small brinjals with a plus cut from bottom leaving stems intact and soak in salted water|Grind roasted peanuts dry coconut goda masala red chilli powder ginger garlic paste jaggery and salt into coarse stuffing paste|Stuff the spice blend firmly inside slit brinjals reserving remaining stuffing for gravy|Heat oil in a broad pan splutter mustard cumin seeds hing curry leaves and chopped onion until softened|Place stuffed brinjals gently into pan turning occasionally for 4 minutes until outer skin blisters lightly|Pour remaining stuffing with 1 cup water cover pan and simmer on low heat for 15 minutes until brinjals are fork tender",
        "source_1": "https://hebbarskitchen.com/bharli-vangi-recipe-stuffed-brinjal/",
        "source_2": "https://www.vegrecipesofindia.com/bharli-vangi-recipe-stuffed-aubergines/",
        "source_3": "https://www.archanaskitchen.com/bharli-vangi-recipe-maharashtrian-style-stuffed-eggplant",
        "url_checked": "yes",
        "verified": "no",
        "review_flag": "pending_pilot_approval"
    },
    {
        "dish_name": "Kolhapuri Tambda Rassa",
        "romanized_local_name": "Tambda Rassa",
        "native_name": "तांबडा रस्सा",
        "native_name_checked": "yes",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "non-vegetarian",
        "meat_type": "mutton",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_minutes": 45,
        "servings": 3,
        "main_ingredients": "mutton|onion|coconut|ginger|garlic",
        "pantry_items": "oil|bay leaf|cloves|cinnamon|black pepper|turmeric|kashmiri red chilli powder|kolhapuri masala|sesame seeds|poppy seeds|salt|coriander leaves",
        "ingredients_with_quantity": "500 g mutton|2 medium onion|3 tbsp dry coconut|1.5 tbsp ginger garlic paste|3 tbsp oil|1 bay leaf|3 cloves|1 inch cinnamon|1 tsp black pepper|1/2 tsp turmeric|1.5 tsp kashmiri red chilli powder|2 tsp kolhapuri masala|1 tsp sesame seeds|1 tsp poppy seeds|salt to taste|3.5 cup water|2 tbsp coriander leaves",
        "steps": "Pressure cook washed mutton pieces with chopped onion ginger garlic paste turmeric salt and 3.5 cups water for 5 whistles to extract clear meat stock|Roast one sliced onion on flame dry roast coconut sesame seeds poppy seeds and whole spices then grind with soaked red chillies into ultra smooth paste|Heat oil in a deep pot crackle bay leaf and fry ground red spice paste on medium flame until oil separates|Add kashmiri chilli powder and authentic kolhapuri kanda lasun masala sauteing for 1 minute without scorching|Strain and pour boiling hot mutton stock into pot along with cooked mutton pieces stirring thoroughly|Simmer rassa on gentle heat for 10 minutes until deep red aromatic oil floating tarri layer forms on surface then garnish with coriander",
        "source_1": "https://www.sanjeevkapoor.com/Recipe/Kolhapuri-Tambda-Rassa-Sirke-Wale-Pyaaz-KhaanaKhazana.html",
        "source_2": "https://madhurasrecipe.com/non-veg/kolhapuri-tambda-rassa/",
        "source_3": "https://deccantiffin.co.uk/blogs/recipes/kolhapuri-tambda-rassa",
        "url_checked": "yes",
        "verified": "no",
        "review_flag": "pending_pilot_approval"
    },
    {
        "dish_name": "Surmai Fry",
        "romanized_local_name": "Surmai Rava Fry",
        "native_name": "सुरमई फ्राय",
        "native_name_checked": "yes",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "non-vegetarian",
        "meat_type": "fish",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_minutes": 20,
        "servings": 2,
        "main_ingredients": "fish|rava|ginger|garlic",
        "pantry_items": "oil|turmeric|red chilli powder|lemon juice|coriander powder|cumin powder|salt|coriander leaves",
        "ingredients_with_quantity": "300 g fish|1/2 cup rava|1 tbsp ginger garlic paste|3 tbsp oil|1/2 tsp turmeric|1.5 tbsp red chilli powder|1 tbsp lemon juice|1 tsp coriander powder|1/2 tsp cumin powder|salt to taste|1 tbsp coriander leaves",
        "steps": "Wash kingfish steaks thoroughly pat dry and rub with ginger garlic paste turmeric lemon juice and salt for 15 minutes|Blend red chilli powder coriander cumin powder and chopped cilantro with drops of water into thick spice rub|Coat fish slices evenly with spice paste on both sides|Mix fine semolina rava with a pinch of salt and red chilli powder in a shallow plate|Press marinated steaks into rava coating firmly on all sides and shake off excess semolina|Shallow fry on hot tava in medium oil for 3 to 4 minutes per side until crust turns golden crunchy and fish is cooked through",
        "source_1": "https://www.archanaskitchen.com/malvani-surmai-fry-recipe-seer-fish-fry",
        "source_2": "https://hebbarskitchen.com/surmai-fish-fry-recipe-seer-fish-fry/",
        "source_3": "https://kunalsfoodjourney.com/crispy-surmai-rava-fry-recipe-malvani-style/",
        "url_checked": "yes",
        "verified": "no",
        "review_flag": "pending_pilot_approval"
    },
    {
        "dish_name": "Jowar Bhakri",
        "romanized_local_name": "Jowarichi Bhakri",
        "native_name": "ज्वारीची भाकरी",
        "native_name_checked": "yes",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_minutes": 20,
        "servings": 2,
        "main_ingredients": "jowar|water",
        "pantry_items": "salt|ghee",
        "ingredients_with_quantity": "1.5 cup jowar|1 cup water|1 pinch salt|1 tsp ghee",
        "steps": "Take freshly milled sorghum flour in a wide flat parat plate making a well in center|Pour hot boiling water gradually into flour mixing with a wooden spoon until dough clumps form|Knead vigorously with palms and heel of hand for 3 minutes adding warm water drops until soft crack free dough forms|Divide into smooth portions and dust working platform generously with dry jowar flour|Pat dough ball flat with palm in circular motion rotating continuously to expand into an even round disc|Place on hot dry iron tawa patted side up moisten top surface with wet hand then flip once water evaporates and roast until puffed",
        "source_1": "https://hebbarskitchen.com/jowar-roti-recipe-jowar-bhakri/",
        "source_2": "https://www.vegrecipesofindia.com/jowar-roti-recipe/",
        "source_3": "https://www.archanaskitchen.com/jowar-bhakri-recipe-sorghum-flatbread",
        "url_checked": "yes",
        "verified": "no",
        "review_flag": "pending_pilot_approval"
    },
    {
        "dish_name": "Masale Bhaat",
        "romanized_local_name": "Masale Bhaat",
        "native_name": "मसाले भात",
        "native_name_checked": "yes",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "diet": "vegetarian",
        "meat_type": "none",
        "contains_egg": "no",
        "pan_india": "yes",
        "time_minutes": 30,
        "servings": 3,
        "main_ingredients": "rice|eggplant|green peas|coconut",
        "pantry_items": "ghee|oil|mustard seeds|cumin seeds|hing|curry leaves|green chilli|turmeric|goda masala|salt|jaggery|coriander leaves|lemon juice",
        "ingredients_with_quantity": "1 cup rice|1 cup eggplant|1/2 cup green peas|3 tbsp coconut|1 tbsp ghee|1 tbsp oil|1 tsp mustard seeds|1 tsp cumin seeds|1 pinch hing|8 curry leaves|2 green chilli|1/2 tsp turmeric|2.5 tsp goda masala|salt to taste|1/2 tsp jaggery|2 cup water|1 tbsp lemon juice|2 tbsp coriander leaves",
        "steps": "Rinse short grain rice thoroughly and soak in water for 20 minutes then drain completely|Heat oil and ghee together in a heavy bottomed pot crackle mustard seeds cumin seeds hing and curry leaves with slit green chillies|Add diced eggplants green peas turmeric and goda masala sauteing for 3 minutes until fragrant|Add drained rice to pot gently tossing with spices and vegetables for 2 minutes|Pour 2 cups hot water add salt and jaggery then bring to rolling boil|Cover tightly with lid simmer on low heat for 12 minutes until water absorbs turn off flame drizzle lemon juice and garnish with fresh coconut and coriander",
        "source_1": "https://www.vegrecipesofindia.com/masale-bhaat-recipe-maharashtrian-masale-bhaat/",
        "source_2": "https://hebbarskitchen.com/masale-bhaat-recipe-masala-rice/",
        "source_3": "https://www.archanaskitchen.com/traditional-maharashtrian-masale-baath-spiced-vegetable-rice-recipe",
        "url_checked": "yes",
        "verified": "no",
        "review_flag": "pending_pilot_approval"
    }
]

def update_sources():
    # Read existing sources
    rows = []
    with open(SOURCES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Pilot dish name lookup
    pilot_map = {p["dish_name"]: p for p in PILOT_DISHES}

    updated_rows = []
    for r in rows:
        d_name = r["dish_name"]
        if d_name in pilot_map:
            p = pilot_map[d_name]
            updated_rows.append({
                "dish_name": p["dish_name"],
                "romanized_local_name": p["romanized_local_name"],
                "native_name": p["native_name"],
                "native_name_checked": "yes",
                "state": p["state"],
                "tier": p["tier"],
                "diet": p["diet"],
                "meat_type": p["meat_type"],
                "contains_egg": p["contains_egg"],
                "pan_india": p["pan_india"],
                "source_1": p["source_1"],
                "source_2": p["source_2"],
                "source_3": p["source_3"],
                "url_checked": "yes",
                "verified": "no",
                "review_flag": "pending_pilot_approval"
            })
        else:
            # Unverified catalogue dish: clear heuristic URLs, set url_checked='no'
            diet = r.get("diet", "vegetarian")
            m_type = "none"
            if "fish" in d_name.lower() or "surmai" in d_name.lower() or "bombil" in d_name.lower() or "kolambi" in d_name.lower() or "prawn" in d_name.lower():
                m_type = "fish"
            elif "chicken" in d_name.lower() or "kombdi" in d_name.lower() or "murgh" in d_name.lower():
                m_type = "chicken"
            elif "mutton" in d_name.lower() or "gosht" in d_name.lower() or "sukka" in d_name.lower():
                m_type = "mutton"
            elif "pork" in d_name.lower() or "choris" in d_name.lower() or "sorpotel" in d_name.lower():
                m_type = "pork"

            updated_rows.append({
                "dish_name": d_name,
                "romanized_local_name": d_name,
                "native_name": "", # Cleared: native script only copied from verified pages
                "native_name_checked": "no",
                "state": r.get("state", "Maharashtra"),
                "tier": r.get("tier", "Tier 1"),
                "diet": diet,
                "meat_type": m_type,
                "contains_egg": "yes" if "egg" in d_name.lower() or "anda" in d_name.lower() else "no",
                "pan_india": r.get("pan_india", "no"),
                "source_1": "",
                "source_2": "",
                "source_3": "",
                "url_checked": "no",
                "verified": "no",
                "review_flag": "pending"
            })

    # Save maharashtra_sources.csv with new schema
    fields = [
        "dish_name", "romanized_local_name", "native_name", "native_name_checked",
        "state", "tier", "diet", "meat_type", "contains_egg", "pan_india",
        "source_1", "source_2", "source_3", "url_checked", "verified", "review_flag"
    ]
    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(updated_rows)

    # Write data/raw/maharashtra_drafts.csv with full recipe fields
    draft_fields = [
        "dish_name", "romanized_local_name", "native_name", "native_name_checked",
        "state", "tier", "diet", "meat_type", "contains_egg", "pan_india",
        "time_minutes", "servings", "main_ingredients", "pantry_items",
        "ingredients_with_quantity", "steps",
        "source_1", "source_2", "source_3", "url_checked", "verified", "review_flag"
    ]
    with open(DRAFTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=draft_fields)
        writer.writeheader()
        writer.writerows(PILOT_DISHES)

    print(f"Updated {SOURCES_CSV} with verified schema and cleaned URLs.")
    print(f"Generated {len(PILOT_DISHES)} pilot dishes into {DRAFTS_CSV}.")

if __name__ == "__main__":
    update_sources()

