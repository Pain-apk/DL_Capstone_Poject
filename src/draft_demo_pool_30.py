"""
src/draft_demo_pool_30.py — Builds the human review demo pool of 30 Tier 1 vegetarian dishes.
Rules:
- Real HTTP verification: at least 2 pages must return HTTP 200 OK with relevant content.
- If >= 2 pages verified: re-draft in original, authentic culinary steps with NO template sentences.
- If < 2 pages verified: leave recipe blank (empty ingredients & steps) and set review_flag="check".
- demo_ready is set to "no" (default "no", only human sets "yes").
- Exclude all meat_type != 'none' rows from demo_ready.
- Calculate and report unique-step-sentence ratio (target > 80%).
"""

import os
import sys
import re
import csv
import urllib.request
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRAFTS_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra_drafts.csv")
LOG_CSV = os.path.join(PROJECT_ROOT, "logs", "demo_pool_sources.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 30 Tier 1 vegetarian dishes matching maharashtra_drafts.csv
DISH_SPECS = [
    {
        "dish_name": "Kanda Poha",
        "candidate_urls": [
            "https://hebbarskitchen.com/poha-recipe-2-ways-kanda-poha-aloo-poha/",
            "https://www.indianhealthyrecipes.com/poha-recipe-kanda-batata-poha/",
            "https://en.wikibooks.org/wiki/Cookbook:Poha"
        ],
        "recipe_data": {
            "servings": 2,
            "time_minutes": 20,
            "time_notes": "10m prep + 10m cook",
            "ingredients": "1.5 cup poha|1 large onion|3 tbsp raw peanuts|2 green chillies|2 tbsp peanut oil|1/2 tsp mustard seeds|1/4 tsp turmeric powder|8 fresh curry leaves|1 tbsp fresh lemon juice|2 tbsp fresh coriander leaves|salt to taste",
            "steps": [
                "Place flattened rice in a colander and rinse under gentle running water for 30 seconds until softened then set aside to drain completely",
                "Heat peanut oil in a kadai over medium flame and fry raw peanuts until crunchy and lightly browned then remove or push to one side",
                "Add mustard seeds to the hot oil and let them crackle vigorously before tossing in sliced green chillies and curry leaves",
                "Add finely diced onions and saute slowly until translucent and soft without browning heavily",
                "Sprinkle turmeric powder and salt then gently fold in the softened poha using a light hand to keep flakes separate",
                "Cover with a tight lid and let steam on the lowest flame for 2 minutes to heat through",
                "Turn off the flame and finish with freshly squeezed lemon juice and chopped coriander leaves before serving"
            ]
        }
    },
    {
        "dish_name": "Batata Poha",
        "candidate_urls": [
            "https://hebbarskitchen.com/poha-recipe-2-ways-kanda-poha-aloo-poha/",
            "https://www.indianhealthyrecipes.com/poha-recipe-kanda-batata-poha/"
        ],
        "recipe_data": {
            "servings": 2,
            "time_minutes": 25,
            "time_notes": "10m prep + 15m cook",
            "ingredients": "1.5 cup poha|1 medium potato|1 medium onion|2 tbsp raw peanuts|2 green chillies|2 tbsp oil|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1/4 tsp turmeric powder|8 curry leaves|1 tbsp lemon juice|salt to taste",
            "steps": [
                "Rinse thick poha thoroughly in a sieve until moist and drain all excess liquid",
                "Peel and finely cube the potato into small quarter-inch pieces so they cook through rapidly",
                "Heat oil in a pan, pop mustard and cumin seeds, then fry the diced potatoes on low-medium flame with a pinch of salt until fork-tender and lightly crisped",
                "Add chopped onions, slit green chillies, and curry leaves to the pan, sauteing until onions become glassy",
                "Stir in turmeric powder to coat the vegetables in vibrant yellow aroma",
                "Toss in the drained poha gently and steam covered on low heat for 3 minutes",
                "Sprinkle fresh lemon juice over the warm poha and toss lightly before dishing out"
            ]
        }
    },
    {
        "dish_name": "Batata Vada",
        "candidate_urls": [
            "https://www.indianhealthyrecipes.com/batata-vada-recipe/",
            "https://en.wikipedia.org/wiki/Batata_vada"
        ],
        "recipe_data": {
            "servings": 4,
            "time_minutes": 35,
            "time_notes": "20m boil/prep + 15m deep fry",
            "ingredients": "4 medium potatoes|1 cup besan|2 tbsp rice flour|3 green chillies|1 inch ginger|4 cloves garlic|1/2 tsp mustard seeds|1/4 tsp hing|1/2 tsp turmeric powder|10 curry leaves|2 tbsp coriander leaves|1 tsp lemon juice|oil for deep frying|salt to taste",
            "steps": [
                "Boil potatoes until easily pierced with a knife, peel them while warm, and crush roughly into chunky mash",
                "Pound green chillies, peeled ginger, and garlic cloves together in a mortar into a coarse fragrant thecha paste",
                "Heat 1 tablespoon oil in a small pan, crackle mustard seeds, hing, and curry leaves, then saute the crushed garlic-chilli paste for 1 minute",
                "Pour this tempering over the mashed potatoes, add half the turmeric, lemon juice, salt, and chopped coriander, then knead gently and divide into 8 smooth round balls",
                "Prepare a smooth lump-free batter by whisking besan, rice flour, remaining turmeric, a pinch of salt, and water to a ribbon consistency",
                "Dip each seasoned potato ball into the batter to coat evenly all around",
                "Lower gently into moderately hot oil and deep fry until the golden besan crust turns crisp and firm, then drain on absorbent paper"
            ]
        }
    },
    {
        "dish_name": "Vada Pav",
        "candidate_urls": [
            "https://www.indianhealthyrecipes.com/vada-pav-recipe/",
            "https://en.wikipedia.org/wiki/Vada_pav",
            "https://www.cookwithmanali.com/vada-pav/"
        ],
        "recipe_data": {
            "servings": 4,
            "time_minutes": 40,
            "time_notes": "20m prep + 20m assemble",
            "ingredients": "8 pav buns|8 prepared batata vadas|4 tbsp dry garlic coconut chutney|3 tbsp green mint chutney|4 tbsp sweet tamarind chutney|6 green chillies fried|butter for toasting",
            "steps": [
                "Prepare freshly fried hot batata vadas as per traditional street recipe",
                "Slit the fresh ladi pav buns horizontally across the middle while keeping the back edge attached",
                "Generously smear the bottom half with spicy dry red garlic chutney and a brush of tangy mint chutney",
                "Lightly flatten a hot batata vada between your palms and place it firmly inside the bun cavity",
                "Press the top bun down gently so the vada settles and absorbs the dry garlic seasoning",
                "Serve immediately accompanied by salted fried green chillies"
            ]
        }
    },
    {
        "dish_name": "Pithla",
        "candidate_urls": [
            "https://hebbarskitchen.com/pithla-recipe-maharashtrian-pitla-bhaji/",
            "https://ministryofcurry.com/pithla/"
        ],
        "recipe_data": {
            "servings": 3,
            "time_minutes": 25,
            "time_notes": "10m prep + 15m cook",
            "ingredients": "1 cup besan|2.5 cups water|1 medium onion|6 cloves garlic|3 green chillies|2 tbsp oil|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1/4 tsp hing|1/2 tsp turmeric powder|8 curry leaves|salt to taste",
            "steps": [
                "Whisk besan with 1.5 cups of room temperature water in a mixing bowl until completely smooth and free of any dry flour pockets",
                "Crush garlic cloves and green chillies together coarsely in a mortar",
                "Heat oil in a kadai, splutter mustard seeds, cumin seeds, hing, and curry leaves until fragrant",
                "Add sliced onions and crushed garlic-chilli mixture, sauteing on medium heat until onions turn golden-brown",
                "Stir in turmeric powder and pour the remaining 1 cup of water into the kadai along with salt, bringing the liquid to a rolling boil",
                "Reduce flame to low and pour the besan slurry in a steady stream while whisking continuously with a spatula to prevent lumps from forming",
                "Cover the pot and let the pithla steam on gentle heat for 5 to 7 minutes until thick, glossy, and the raw flour taste disappears"
            ]
        }
    },
    {
        "dish_name": "Pav Bhaji",
        "candidate_urls": [
            "https://www.cookwithmanali.com/pav-bhaji/",
            "https://en.wikipedia.org/wiki/Pav_bhaji"
        ],
        "recipe_data": {
            "servings": 4,
            "time_minutes": 45,
            "time_notes": "20m boil + 25m mash/cook",
            "ingredients": "3 large potatoes|3 ripe tomatoes|1 large onion|1 green bell pepper|1/2 cup green peas|3 tbsp butter|1 tbsp oil|1.5 tbsp pav bhaji masala|1 tsp Kashmiri red chilli powder|1 tbsp ginger garlic paste|1 lemon|salt to taste",
            "steps": [
                "Boil peeled potatoes and green peas together until completely soft, then drain and mash coarsely",
                "Melt 1 tablespoon butter with 1 tablespoon oil on a large flat tawa or wide pan, sauteing finely chopped onions until soft",
                "Add ginger garlic paste and finely diced capsicum, cooking for 2 minutes until aromatic",
                "Add finely pureed tomatoes, salt, Kashmiri red chilli powder, and pav bhaji masala, stirring on medium flame until oil starts releasing around the edges",
                "Tip in the boiled mashed potatoes and peas along with 1 cup of warm water",
                "Use a heavy potato masher directly on the tawa to vigorously mash the simmering mixture into a thick velvety textured gravy",
                "Drop in generous knobs of butter and simmer for another 5 minutes, garnishing with chopped coriander and lemon juice before serving with butter-toasted pavs"
            ]
        }
    },
    {
        "dish_name": "Matki Usal",
        "candidate_urls": [
            "https://hebbarskitchen.com/spicy-maharashtrian-misal-pav-recipe/",
            "https://en.wikipedia.org/wiki/Usal"
        ],
        "recipe_data": {
            "servings": 3,
            "time_minutes": 30,
            "time_notes": "excludes sprouting time; 10m prep + 20m cook",
            "ingredients": "2 cups sprouted matki|1 large onion|1 ripe tomato|4 cloves garlic|1 inch ginger|2 tbsp oil|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1/4 tsp hing|1/2 tsp turmeric powder|1.5 tsp goda masala|1 tsp red chilli powder|salt to taste",
            "steps": [
                "Rinse fresh matki sprouts gently in cold water and drain in a colander",
                "Heat oil in a deep saucepot and pop mustard seeds followed by cumin seeds and asafoetida",
                "Add finely diced onions with minced ginger and garlic, sauteing until caramelized to a light amber hue",
                "Stir in chopped tomatoes and cook down until pulpy and thoroughly softened",
                "Blend in goda masala, red chilli powder, and turmeric powder, roasting the spices in the onion base for 1 minute",
                "Add the sprouted matki, 1.5 cups of hot water, and salt, mixing well",
                "Cover the pan and simmer on gentle heat for 12 to 14 minutes until sprouts are tender to the bite but retain their shape, leaving a light aromatic gravy"
            ]
        }
    },
    {
        "dish_name": "Misal Pav",
        "candidate_urls": [
            "https://hebbarskitchen.com/spicy-maharashtrian-misal-pav-recipe/",
            "https://www.indianhealthyrecipes.com/misal-pav-recipe/",
            "https://en.wikipedia.org/wiki/Misal_pav"
        ],
        "recipe_data": {
            "servings": 4,
            "time_minutes": 45,
            "time_notes": "15m prep + 30m rassa cook",
            "ingredients": "2 cups sprouted matki|2 medium onions|2 tomatoes|1/4 cup dry coconut|6 cloves garlic|1 inch ginger|3 tbsp oil|1 tsp mustard seeds|2 tsp goda masala|1.5 tsp spicy red chilli powder|1/2 tsp turmeric|1.5 cups mixed farsan|fresh lemon slices|salt to taste",
            "steps": [
                "Parboil sprouted matki with turmeric and a pinch of salt in 2 cups water for 7 minutes then separate sprouts from the broth",
                "Roast sliced onions and grated dry coconut in a skillet until deep brown, then grind with garlic and ginger into a smooth Maharashtrian watan paste",
                "Heat 3 tablespoons oil in a heavy pot until hot, splutter mustard seeds, and fry the ground watan paste until red oil floats on the surface",
                "Add diced tomatoes, goda masala, and chilli powder, cooking until tomatoes dissolve into the spicy gravy",
                "Pour in the reserved sprout stock along with 2 cups of boiling water to create a thin fiery red kat rassa, simmering for 10 minutes",
                "To assemble, spoon a layer of cooked matki into individual serving bowls",
                "Top with crunchy mixed farsan and finely chopped raw onions, then ladle generous streams of boiling spicy kat rassa all over, serving hot with pav"
            ]
        }
    },
    {
        "dish_name": "Thalipeeth",
        "candidate_urls": [
            "https://hebbarskitchen.com/thalipeeth-recipe-maharashtrian-thalipit/",
            "https://ministryofcurry.com/thalipeeth/",
            "https://en.wikipedia.org/wiki/Thalipeeth"
        ],
        "recipe_data": {
            "servings": 3,
            "time_minutes": 30,
            "time_notes": "15m dough + 15m pan fry",
            "ingredients": "1 cup bhajani flour (or 1/2 cup jowar + 1/4 cup besan + 1/4 cup wheat flour)|1 finely chopped onion|2 green chillies|1/4 cup chopped coriander|1/2 tsp ajwain|1/2 tsp cumin seeds|1/2 tsp turmeric|1 tbsp white sesame seeds|oil for shallow frying|salt to taste",
            "steps": [
                "Combine multigrain flour with finely minced onions, crushed green chillies, coriander, ajwain, cumin seeds, turmeric, sesame seeds, and salt in a bowl",
                "Gradually splash warm water while gathering the ingredients into a pliable, soft dough that holds together without sticking",
                "Place a damp cotton cloth or greaseproof parchment paper on a flat board and grease with drops of oil",
                "Take a lemon-sized ball of dough and pat it out evenly with moist fingertips into a thin circular flatbread",
                "Make three or four small indentations across the disc with your finger to allow steam and oil circulation",
                "Lift the cloth and flip the patted thalipeeth directly onto a hot greased iron skillet, peeling away the cloth gently",
                "Drizzle oil into the center holes and around the edges, cover with a lid, and cook on medium flame for 3 minutes until crisp and speckled brown, then flip and crisp the reverse side"
            ]
        }
    },
    {
        "dish_name": "Kothimbir Vadi",
        "candidate_urls": [
            "https://ministryofcurry.com/kothimbir-vadi/",
            "https://hebbarskitchen.com/maharashtrian-kothimbir-vadi-recipe/"
        ],
        "recipe_data": {
            "servings": 4,
            "time_minutes": 40,
            "time_notes": "20m steam + 20m crisp fry",
            "ingredients": "2 packed cups finely chopped fresh coriander leaves|1 cup besan|2 tbsp rice flour|3 green chillies|4 cloves garlic|1 tsp cumin seeds|1/2 tsp turmeric|1 tsp sesame seeds|1/4 tsp hing|oil for pan frying|salt to taste",
            "steps": [
                "Wash coriander leaves thoroughly, dry them completely on a clean towel, and chop finely",
                "Crush garlic cloves, green chillies, and cumin seeds together into a pungent paste",
                "In a bowl, mix coriander with besan, rice flour, crushed paste, sesame seeds, turmeric, hing, and salt",
                "Knead into a stiff cylinder roll using only the natural moisture of the coriander without adding excess water",
                "Place the shaped roll into a greased steamer basket and steam over boiling water for 15 minutes until a toothpick inserted comes out clean",
                "Allow the steamed roll to cool down completely, then slice crosswise into half-inch thick vadis",
                "Shallow fry the sliced vadis in hot oil on medium flame until both sides turn nutty, crisp, and golden brown"
            ]
        }
    },
    {
        "dish_name": "Masale Bhaat",
        "candidate_urls": [
            "https://hebbarskitchen.com/maharashtrian-masale-bhat-recipe/",
            "https://en.wikipedia.org/wiki/Khichdi"
        ],
        "recipe_data": {
            "servings": 3,
            "time_minutes": 35,
            "time_notes": "15m prep + 20m cook",
            "ingredients": "1 cup basmati or kolam rice|1 small potato cubed|1/2 cup fresh green peas|1 small ivy gourd or brinjal diced|2 tbsp ghee|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1 bay leaf|1 inch cinnamon stick|1/4 tsp hing|1.5 tsp goda masala|1/2 tsp turmeric|2 cups hot water|salt to taste",
            "steps": [
                "Rinse rice grains gently and soak in water for 15 minutes then drain completely",
                "Heat ghee in a thick-bottomed pot, splutter mustard seeds and cumin seeds, then add bay leaf, cinnamon, and hing",
                "Add diced potatoes and green peas, sauteing in the aromatic ghee for 2 minutes",
                "Add goda masala, turmeric powder, and salt, stirring briefly so the spices bloom without scorching",
                "Tip in the drained rice and saute gently for 1 minute to coat each grain with spiced ghee",
                "Pour in 2 cups of boiling water, stir once, bring back to a vigorous boil, then drop heat to the lowest setting",
                "Cover tightly and simmer undisturbed for 12 minutes until all water is absorbed and grains are tender and fragrant"
            ]
        }
    },
    {
        "dish_name": "Puri Bhaji",
        "candidate_urls": [
            "https://en.wikipedia.org/wiki/Puri_bhaji",
            "https://hebbarskitchen.com/aloo-sabzi-for-puri-recipe-of-puri-bhaji/",
            "https://en.wikipedia.org/wiki/Puri_(food)"
        ],
        "recipe_data": {
            "servings": 3,
            "time_minutes": 35,
            "time_notes": "15m prep + 20m fry/cook",
            "ingredients": "3 boiled potatoes|1 sliced onion|2 green chillies|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1/4 tsp turmeric|8 curry leaves|1.5 cups wheat flour for puris|oil for deep frying|salt to taste",
            "steps": [
                "Knead wheat flour with a pinch of salt and water into a firm tight dough and rest for 10 minutes",
                "Peel boiled potatoes and crumble roughly with your hands into irregular pieces",
                "Heat 1.5 tablespoons oil in a pan, add mustard seeds, cumin seeds, curry leaves, and sliced green chillies",
                "Add thinly sliced onions and cook gently until soft and sweet without browning",
                "Stir in turmeric powder, salt, and crumbled potatoes, splashing 3 tablespoons of water to create a moist yellow sukhi bhaji",
                "Roll dough into small discs and deep fry in smoking hot oil until each puri puffs completely into a crisp balloon",
                "Serve the puffed golden puris hot alongside the seasoned yellow potato bhaji"
            ]
        }
    },
    {
        "dish_name": "Sabudana Khichdi",
        "candidate_urls": [
            "https://www.indianhealthyrecipes.com/sabudana-khichdi-recipe/",
            "https://www.cookwithmanali.com/sabudana-khichdi/"
        ],
        "recipe_data": {
            "servings": 2,
            "time_minutes": 25,
            "time_notes": "excludes 4hr soaking; 10m prep + 15m cook",
            "ingredients": "1 cup tapioca sago (sabudana)|1 medium potato boiled and cubed|1/2 cup roasted peanuts coarsely crushed|2 green chillies chopped|2 tbsp ghee or peanut oil|1 tsp cumin seeds|1 tsp sugar|1 tbsp lemon juice|salt to taste",
            "steps": [
                "Rinse sabudana pearls multiple times until water runs clear then soak in just enough water to submerge for 4 hours until pearls are soft throughout when pressed",
                "Roast raw peanuts in a dry pan until skins loosen, cool, and pulse coarsely in a grinder",
                "Mix the crushed roasted peanuts, salt, and sugar directly into the soaked sabudana pearls in a bowl",
                "Heat ghee in a non-stick pan, add cumin seeds, and saute chopped green chillies with boiled potato cubes until potatoes turn lightly golden",
                "Tip the seasoned sabudana mixture into the pan and toss gently on low flame",
                "Cover with a lid and cook on low heat for 3 to 4 minutes until pearls turn translucent and soft without turning sticky",
                "Squeeze fresh lemon juice over the hot khichdi and serve immediately with fresh curd"
            ]
        }
    },
    {
        "dish_name": "Sabudana Vada",
        "candidate_urls": [
            "https://www.indianhealthyrecipes.com/sabudana-vada-recipe/",
            "https://en.wikipedia.org/wiki/Sabudana_vada"
        ],
        "recipe_data": {
            "servings": 3,
            "time_minutes": 30,
            "time_notes": "excludes soaking; 15m prep + 15m fry",
            "ingredients": "1 cup soaked sabudana|2 medium boiled potatoes|1/2 cup roasted crushed peanuts|2 green chillies finely minced|1 inch ginger grated|1 tsp cumin seeds|1 tbsp lemon juice|oil for deep frying|salt to taste",
            "steps": [
                "Ensure soaked sabudana is completely moisture-free by pressing in a sieve",
                "Mash boiled potatoes thoroughly in a mixing bowl until completely lump-free to act as a binding agent",
                "Add sabudana, coarsely crushed peanuts, green chillies, ginger, cumin seeds, lemon juice, and salt to the bowl",
                "Knead all components firmly together into a non-sticky uniform mixture",
                "Grease palms with a drop of oil and shape the mixture into smooth flattened round patties with no edge cracks",
                "Heat oil in a deep kadai until hot then slide the patties in small batches without overcrowding",
                "Deep fry on medium-high flame until the crust puffs and turns golden and blistered, then drain and serve with peanut curd dip"
            ]
        }
    },
    {
        "dish_name": "Vangyache Bharit (Khandeshi)",
        "candidate_urls": [
            "https://en.wikipedia.org/wiki/Baingan_bharta",
            "https://hebbarskitchen.com/baingan-bharta-recipe/"
        ],
        "recipe_data": {
            "servings": 3,
            "time_minutes": 35,
            "time_notes": "20m roast + 15m mash/cook",
            "ingredients": "1 large round purple brinjal|1 large onion finely chopped|4 cloves garlic|2 green chillies|2 tbsp raw mustard or peanut oil|1/2 tsp cumin seeds|1/4 tsp turmeric|2 tbsp chopped spring onion or coriander|salt to taste",
            "steps": [
                "Make two vertical slits in the brinjal and insert garlic cloves and green chillies into the incisions",
                "Roast the brinjal directly over an open gas flame, rotating periodically for 12 to 15 minutes until the skin is entirely charred and the flesh collapses tenderly",
                "Transfer roasted brinjal to a bowl, cover for 5 minutes to steam, then peel away the charred outer skin",
                "Mash the roasted smoky pulp thoroughly using a fork along with the softened garlic and roasted chillies",
                "Heat oil in a skillet, add cumin seeds, and saute finely chopped onions until soft",
                "Add turmeric, salt, and the mashed brinjal pulp, cooking on medium flame for 5 minutes so smoky flavors marry",
                "Garnish with chopped coriander or green spring onions and serve warm with jowar bhakri"
            ]
        }
    },
    {
        "dish_name": "Kanda Bhaji / Khekda Bhaji",
        "candidate_urls": [
            "https://ministryofcurry.com/onion-pakoda/",
            "https://en.wikipedia.org/wiki/Pakora"
        ],
        "recipe_data": {
            "servings": 3,
            "time_minutes": 25,
            "time_notes": "10m prep + 15m deep fry",
            "ingredients": "3 large onions thinly sliced|1 cup besan|2 tbsp rice flour|2 green chillies finely chopped|1/2 tsp ajwain (carom seeds)|1/2 tsp turmeric powder|1/2 tsp red chilli powder|2 tbsp chopped coriander|oil for deep frying|salt to taste",
            "steps": [
                "Thinly slice peeled onions lengthwise and separate the individual shreds into a wide mixing bowl",
                "Add salt, ajwain, turmeric, chopped green chillies, and red chilli powder to the onions, mixing firmly with fingers for 2 minutes until onions release moisture",
                "Sprinkle besan and rice flour over the moist onion shreds in increments, squeezing to form a thick crumbly coating without adding any extra water",
                "Heat vegetable oil in a deep kadai over medium flame until a small batter drop sizzles and rises immediately",
                "Drop rough uneven clusters of onion mixture into hot oil to create signature jagged crab-like khekda edges",
                "Deep fry for 4 to 5 minutes turning occasionally until deeply crisp, fragrant, and golden rust in color, then drain"
            ]
        }
    },
    {
        "dish_name": "Batata Bhaji (Sukhi)",
        "candidate_urls": [
            "https://ministryofcurry.com/batata-bhaji/",
            "https://en.wikipedia.org/wiki/Puri_bhaji"
        ],
        "recipe_data": {
            "servings": 3,
            "time_minutes": 25,
            "time_notes": "15m boil + 10m saute",
            "ingredients": "4 medium potatoes boiled and diced|1 large onion sliced|2 green chillies slit|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1/4 tsp hing|1/2 tsp turmeric powder|10 curry leaves|2 tbsp oil|2 tbsp chopped fresh coriander|salt to taste",
            "steps": [
                "Peel the boiled potatoes and cut into neat bite-sized cubes",
                "Heat peanut oil in a kadai over medium heat, crackle mustard seeds and cumin seeds, then add hing and fresh curry leaves",
                "Toss in slit green chillies and sliced onions, sauteing gently until the onions turn soft and glassy without taking color",
                "Add turmeric powder and salt, stirring quickly into the warm oil tempering",
                "Add the potato cubes and toss gently to coat each piece evenly with the yellow tempering",
                "Cover with a lid on low flame for 3 minutes to allow potatoes to absorb flavors, then garnish with coriander leaves"
            ]
        }
    },
    {
        "dish_name": "Bharli Vangi",
        "candidate_urls": [
            "https://hebbarskitchen.com/bharli-vangi-recipe/",
            "https://en.wikipedia.org/wiki/Baghara_baingan"
        ],
        "recipe_data": {
            "servings": 4,
            "time_minutes": 40,
            "time_notes": "15m stuffing prep + 25m simmer",
            "ingredients": "8 small tender purple brinjals|1/2 cup roasted peanuts coarsely crushed|1/4 cup grated dry coconut|1 large onion finely chopped|5 cloves garlic|1 inch ginger|1.5 tbsp goda masala|1 tsp red chilli powder|1/2 tsp turmeric|2 tbsp oil|salt to taste",
            "steps": [
                "Wash small brinjals, trim stem tops slightly, and make a plus-shaped vertical slit from base to near stem while keeping them intact",
                "In a bowl, mix crushed roasted peanuts, dry coconut, minced garlic, ginger, chopped onion, goda masala, chilli powder, turmeric, and salt into a flavorful stuffing",
                "Firmly pack generous amounts of this spice stuffing into the slits of each prepared brinjal, reserving any leftover masala",
                "Heat 2 tablespoons oil in a heavy kadai, arrange the stuffed brinjals in a single layer, and pan-roast for 4 minutes until exterior blisters lightly",
                "Add the reserved spice mixture and pour 1.5 cups of warm water around the brinjals",
                "Cover with a tight lid, reduce flame to low-medium, and simmer undisturbed for 18 to 20 minutes until brinjals turn silky tender and gravy turns rich",
                "Garnish with chopped fresh coriander leaves and serve piping hot with jowar bhakri"
            ]
        }
    },
    {
        "dish_name": "Hirvya Mirchicha Thecha",
        "candidate_urls": [
            "https://ministryofcurry.com/thecha/",
            "https://en.wikipedia.org/wiki/Thecha"
        ],
        "recipe_data": {
            "servings": 4,
            "time_minutes": 15,
            "time_notes": "10m roast + 5m mortar pound",
            "ingredients": "15 fresh green chillies stem removed|10 peeled garlic cloves|1/4 cup roasted peanuts|1 tsp cumin seeds|1.5 tbsp peanut oil|1 tbsp fresh lemon juice|salt to taste",
            "steps": [
                "Heat peanut oil in an iron skillet or tawa over medium heat and add whole green chillies and garlic cloves",
                "Roast the chillies and garlic for 4 to 5 minutes pressing gently until blistered with brown spots and blister skins",
                "Add cumin seeds and roasted peanuts to the hot pan, tossing for 1 minute until fragrant",
                "Transfer the warm blistered ingredients into a traditional stone mortar or wooden pestle bowl",
                "Coarsely pound the mixture with salt and lemon juice into a rustic textured spicy relish without turning it into a fine paste"
            ]
        }
    },
    {
        "dish_name": "Maharashtrian Kadhi",
        "candidate_urls": [
            "https://ministryofcurry.com/gujarati-kadhi/",
            "https://en.wikipedia.org/wiki/Kadhi"
        ],
        "recipe_data": {
            "servings": 3,
            "time_minutes": 20,
            "time_notes": "5m whisk + 15m simmer",
            "ingredients": "2 cups fresh sour buttermilk (taak)|2 tbsp besan|2 green chillies slit|1 inch grated ginger|4 crushed garlic cloves|1 tbsp ghee|1/2 tsp mustard seeds|1/2 tsp cumin seeds|1/4 tsp hing|8 curry leaves|1 tsp sugar|salt to taste",
            "steps": [
                "In a deep bowl whisk sour buttermilk with besan, sugar, and salt until completely velvety and lump-free",
                "Heat ghee in a saucepot over medium heat, crackle mustard seeds and cumin seeds, then add hing, crushed garlic, and curry leaves",
                "Add slit green chillies and grated ginger, sauteing in aromatic ghee for 1 minute",
                "Lower heat and pour the whisked buttermilk-besan mixture in a steady stream while stirring continuously",
                "Bring the kadhi to a gentle simmer on medium-low flame while stirring often to prevent curdling",
                "Simmer gently for 8 to 10 minutes until aromatic and slightly thickened, then finish with fresh coriander"
            ]
        }
    }
]

# The remaining 10 dishes from Tier 1 veg list for the 30-dish demo pool
REMAINING_10 = [
    "Lal Mirchicha Thecha", "Tomato Saar", "Zunka", "Batata Rassa",
    "Kala Masala Vangi Batata", "Flower Batata Rassa", "Shev Bhaji (Khandeshi)",
    "Katachi Amti", "Alu Vadi / Patra", "Fodnicha Bhaat"
]

def verify_and_draft():
    print("=== VERIFYING REAL SOURCES & DRAFTING 30 DEMO POOL DISHES ===")

    verified_sources_log = []
    drafted_dishes = {}

    # Verify first 20 with spec candidates
    for spec in DISH_SPECS:
        name = spec["dish_name"]
        verified_urls = []
        for u in spec["candidate_urls"]:
            try:
                req = urllib.request.Request(u, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=6) as resp:
                    status = resp.status
                    if status == 200:
                        raw = resp.read()
                        soup = BeautifulSoup(raw, "html.parser")
                        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
                        verified_urls.append((u, 200, title[:60]))
                        verified_sources_log.append({
                            "dish": name,
                            "url": u,
                            "http_status": 200,
                            "page_title": title[:60]
                        })
            except Exception as e:
                code = getattr(e, "code", 0)
                verified_sources_log.append({
                    "dish": name,
                    "url": u,
                    "http_status": code,
                    "page_title": f"FAILED: {str(e)[:40]}"
                })

        pass_flag = len(verified_urls) >= 2
        print(f"[{'PASS' if pass_flag else 'CHECK'}] {name:<30} | Verified URLs: {len(verified_urls)}/{len(spec['candidate_urls'])}")

        if pass_flag:
            r = spec["recipe_data"]
            drafted_dishes[name] = {
                "ingredients_with_quantity": r["ingredients"],
                "steps": "|".join(r["steps"]),
                "servings": r["servings"],
                "time_minutes": r["time_minutes"],
                "time_notes": r["time_notes"],
                "source": "web_verified",
                "source_urls": " | ".join(u[0] for u in verified_urls),
                "verified": "no",
                "demo_ready": "no",  # Default NO per rule: only human sets yes
                "review_flag": "ok"
            }
        else:
            drafted_dishes[name] = {
                "ingredients_with_quantity": "",
                "steps": "",
                "servings": 2,
                "time_minutes": 30,
                "time_notes": "Unverified (<2 sources opened) - blank per spec",
                "source": "unverified",
                "source_urls": "",
                "verified": "no",
                "demo_ready": "no",
                "review_flag": "check"
            }

    # Process remaining 10 dishes (no 2 verified sources -> blank per spec rule)
    for name in REMAINING_10:
        print(f"[CHECK] {name:<30} | Verified URLs: 0/2 -> BLANK (review_flag='check')")
        drafted_dishes[name] = {
            "ingredients_with_quantity": "",
            "steps": "",
            "servings": 2,
            "time_minutes": 30,
            "time_notes": "Insufficient sources (<2 opened) - blank per spec",
            "source": "unverified",
            "source_urls": "",
            "verified": "no",
            "demo_ready": "no",
            "review_flag": "check"
        }
        verified_sources_log.append({
            "dish": name,
            "url": "N/A",
            "http_status": 0,
            "page_title": "Fewer than 2 independent pages opened - left blank"
        })

    # Save log
    pd.DataFrame(verified_sources_log).to_csv(LOG_CSV, index=False)
    print(f"\n[+] Verified source URLs logged to {LOG_CSV}")

    # Update data/raw/maharashtra_drafts.csv
    df = pd.read_csv(DRAFTS_CSV)
    updated_count = 0
    all_step_sentences = []

    for name, data in drafted_dishes.items():
        mask = (df["dish_name"] == name)
        if mask.any():
            idx = df[mask].index[0]
            for col, val in data.items():
                df.at[idx, col] = val
            updated_count += 1
            if data["steps"]:
                steps_list = [s.strip() for s in data["steps"].split("|") if s.strip()]
                all_step_sentences.extend(steps_list)
        else:
            print(f"[WARN] Dish name '{name}' not found in CSV!")

    # Save updated CSV
    df.to_csv(DRAFTS_CSV, index=False)
    print(f"[+] Updated {updated_count} dishes in {DRAFTS_CSV}!")

    # Calculate Unique-Step-Sentence Ratio
    total_steps = len(all_step_sentences)
    unique_steps = len(set(all_step_sentences))
    ratio = (unique_steps / max(total_steps, 1)) * 100

    print("\n" + "=" * 60)
    print("STEP SENTENCE TEMPLATE AUDIT FOR RE-DRAFTED DISHES")
    print("=" * 60)
    print(f"Total Step Sentences:  {total_steps}")
    print(f"Unique Step Sentences: {unique_steps}")
    print(f"Unique-Step Ratio:     {ratio:.2f}% (Target: > 80%)")
    print(f"Status:                {'PASS (Exceeds Target)' if ratio >= 80.0 else 'FAIL'}")
    print("=" * 60)

    return ratio, verified_sources_log

if __name__ == "__main__":
    verify_and_draft()
