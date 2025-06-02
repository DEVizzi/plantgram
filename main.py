import os
import random
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from instagrapi import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import pickle
from langchain.schema import SystemMessage, HumanMessage
from instagrapi import Client
import cloudinary
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import cloudinary
import cloudinary.uploader
from datetime import datetime
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Instagram Graph API credentials
ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN")  # Long-lived Instagram Graph API token
IG_USER_ID = os.getenv("IG_USER_ID")        # Your Instagram Business Account ID

# Initialize Gemini Chat Model with a supported model name
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.7
)


# List of popular plants
PLANT_LIST = [
    "Monstera deliciosa", "Fiddle Leaf Fig", "Snake Plant", "Peace Lily",
    "Spider", "ZZ Plant", "Pothos Plant", "Rubber Plant",
    "Philodendron", "Succulent", "Aloe Vera",
    "Chinese Evergreen", "Dieffenbachia", "Boston Fern", "Areca Palm",
    "Bamboo Palm", "Dracaena Marginata", "Calathea", "Croton",
    "Jade", "English Ivy", "Anthurium", "Kalanchoe",
    "Begonia", "Cast Iron", "Prayer", "Schefflera",
    "Asparagus Fern", "Peperomia", "Aglaonema", "Hoya",
    "String of Pearls", "String of Hearts", "Oxalis", "Air",
    "Moth Orchid", "Nerve", "Lucky Bamboo", "Money",
    "Cactus", "Swiss Cheese", "Tropical Pitcher",
    "African Violet", "Parlor Palm", "Maidenhair Fern",
    "Silver Satin Pothos", "Echeveria", "Yucca", "Sansevieria Cylindrica",
    "Variegated Monstera", "Alocasia Polly", "Raven ZZ", "Fatsia Japonica",
    "Lipstick", "Dischidia", "Tradescantia", "Banana",
    "Elephant Ear", "Tillandsia Xerographica", "Staghorn Fern",
    "Homalomena", "Caladium", "Epipremnum Cebu Blue", "Philodendron Pink Princess",
    "Philodendron Gloriosum", "Philodendron Melanochrysum",
    "Monstera Adansonii", "Syngonium", "Senecio", "Tacca Chantrieri (Bat)",
    "Bonsai Tree", "Lithops (Living Stones)", "Maranta Leuconeura (Red Prayer)",
    "Coleus", "Dumb Cane", "Mimosa Pudica (Sensitive)",
    "String of Dolphins", "String of Bananas", "Senecio Rowleyanus",
    "Rhipsalis", "Velvet Philodendron", "Scindapsus Pictus", "Pilea Peperomioides",
    "Pilea Moon Valley", "Aeschynanthus", "Nephrolepis exaltata", "Aralia",
    "Clusia Rosea", "Zebra", "Moon Cactus", "Bromeliad", "Torch Ginger",
    "Sago Palm", "Kentia Palm", "Lady Palm", "Norfolk Island Pine",
    "Ponytail Palm", "Umbrella Tree", "Norfolk Pine", "Bird’s Nest Fern",
    "Bleeding Heart", "Coral Cactus", "Creeping Jenny", "Crown of Thorns",
    "Desert Rose", "Fatshedera", "Flamingo Flower", "Foxtail Fern",
    "Friendship", "Ghost", "Gloxinia", "Golden Barrel Cactus",
    "Golden Pothos", "Hardy Hibiscus", "Hibiscus", "Ice",
    "Indian Laurel Fig", "Kangaroo Paw", "Kangaroo Vine", "Kumquat Tree",
    "Mexican Feather Grass", "Mexican Snowball", "Night Blooming Jasmine",
    "Oleander", "Panda", "Passion Flower", "Persian Shield",
    "Red Edged Dracaena", "Rosary Vine", "Royal Poinciana",
    "Sacred Lotus", "Sweetheart Hoya", "Ti", "Traveler’s Palm",
    "Trumpet Vine", "Vinca", "Watermelon Peperomia", "Weeping Fig",
    "White Bird of Paradise", "White Jasmine", "Wild Ginger", "Wisteria"
]

HOOK_LIST = [
    "Photosynthesizing my way through life ☀️🌿",
    "Living that chlorophyll dream 💚",
    "Thorns and all, still beautiful 🌵✨",
    "Leaf goals, am I right? 🍃😎",
    "Branching into new vibes 🌱🎉",
    "Got that foliage flex on point 🌿💪",
    "Sprouting good energy only 🌱⚡",
    "Rooted in positivity 🌳🌟",
    "Succ-cess looks succulent 🌵😉",
    "Stalk market soaring 📈🌿",
    "Hanging out with my leafy crew 🍃👯",
    "Leaf it to me to brighten your feed 🌿📸",
    "Just here, chlorophyll-ing out 😌🌱",
    "Plant parenthood level: expert 🪴🏆",
    "Water you waiting for? 🌊🌵",
    "Sow it goes 🌱✨",
    "Feeling unbe-leaf-ably good today 🍃😄",
    "Let’s stick together, like vines 🌿🤝",
    "Growing through what I'm going through 🌱💚",
    "Feeling frond-tastic 🌴😎",
    "Every day is a fresh bloom 🌸🌞",
    "Planting dreams and watching them grow 🌱🌟",
    "Vibing at my own photosynthesis frequency 🌞🍃",
    "Moss definitely having a great day 🌿😉",
    "Flourishing like a wildflower 🌼🌻",
    "Still blooming, even in the shade 🌸🌙",
    "Rooting for you always 🌱🙌",
    "Stay wild, plant child 🌿🌼",
    "Turn over a new leaf 🍃🔄",
    "Life's better with a little dirt under your nails 🌱👐",
    "From tiny seeds grow mighty trees 🌳✨",
    "I'm kind of a big dill 🌿🥒",
    "Taking life one bloom at a time 🌸🛤️",
    "Sprouting smiles everywhere I go 🌱😊",
    "Deep roots, strong vibes 🌳⚡",
    "Lettuce turnip the beet 🌱🎶",
    "Potted and thriving 🪴💥",
    "Photosynthesizing my way through the chaos ☀️🌿",
    "Aloe you vera much 🌵💚",
    "Grow through what you go through 🌱💪",
    "More plant, less rant 🌿😌",
    "Living la vida leaf-a 🍃🎶",
    "Succulent state of mind 🌵🧠",
    "Feeling rooted yet reaching for the stars 🌳⭐",
    "Daisy if you do, daisy if you don't 🌼😉",
    "Just a plant trying to photosynthesize its dreams ☀️🌱",
    "Keep calm and grow on 🌿🧘",
    "Sunshine mixed with a little bit of chlorophyll 🌞🍃",
    "Can't stop, won't stop sprouting 🌱💥",
    "Thriving on sunlight and good vibes 🌞🌿",
    "Bloom where you're wildly planted 🌸🌾",
    "Petal to the metal 🌼🚗",
    "Stayin’ sharp like a cactus 🌵⚡",
    "My soil is rich in good energy 🌱✨",
    "Grew it myself, no cap 🧢🌿",
    "This is my plant era 🌱💚",
    "Photosynthesize first, ask questions later 🍃🕶️",
    "Nature's influencer 🌿📱",
    "Budding with excitement 🌱🎉",
    "Making waves in the watering can 🌊🪴",
    "Brb, cultivating my best self 🌱💫",
    "My roots run deep and so do my vibes 🌳✨",
    "Succ it up, buttercup 🌵🌼",
    "In full bloom and feeling it 🌸🔥",
    "Just another day thriving in the jungle 🌴🏝️",
    "Gotta risk it for the biscuit... or the bloom 🌸😉",
    "Living life on the vine 🌿🎢",
    "A little dirt never hurt 🌱👐",
    "Manifesting leafy dreams 🌿🔮",
    "Sprouting joy wherever I grow 🌱😊",
    "Lost in the foliage and loving it 🌿🌀",
    "Sun-chasing and thriving 🌞🍃",
    "Don't stop be-leafing 🌿🎶",
    "Feeling plantastic today 🌱😄",
    "Proud plant parent flex 🪴💪",
    "Natural-born grower 🌱👑",
    "Turning my living room into a jungle one plant at a time 🌴🏡",
    "More blooms, less gloom 🌸😎",
    "Seed by seed, dream by dream 🌱✨",
    "My aesthetic: cozy chaos but make it leafy 🍃🛋️"
]


@tool
def search_plant_image(query: str) -> str:
    """
    Returns a URL of a potted indoor plant image based on the given plant name.
    Tries Unsplash first (HD), falls back to Pexels. Then falls back to generic terms.
    Avoids duplicates using history.txt.
    """

    UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

    if not UNSPLASH_ACCESS_KEY or not PEXELS_API_KEY:
        raise ValueError("Missing API keys for Unsplash or Pexels in environment variables.")

    pot_keywords = {"pot", "potted", "planter", "flowerpot"}
    plant_keywords = {"plant", "leaf", "leaves", "foliage", "greenery", "houseplant", "indoor plant"}
    banned_keywords = {"illustration", "vector", "drawing", "artificial", "render", "3d", "toy", "fake", "vessel"}

    # NEW fallback list
    fallback_terms = [query, "indoor plant", "potted plant", "succulent"]

    def is_valid_metadata(text, is_fallback=False):
        text = text.lower()
        if any(bad in text for bad in banned_keywords):
            return False
        if not is_fallback and not any(good in text for good in pot_keywords):
            return False
        if not any(good in text for good in plant_keywords):
            return False
        return True

    def fetch_unsplash_image(search_term: str, is_fallback=False):
        params = {
            "query": f"{search_term} pot indoor",
            "client_id": UNSPLASH_ACCESS_KEY,
            "per_page": 30,
            "orientation": "portrait",
            "content_filter": "high"
        }
        url = "https://api.unsplash.com/search/photos"
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"Unsplash error: {resp.status_code}")
            return None

        results = resp.json().get("results", [])
        random.shuffle(results)
        for img in results:
            alt = img.get("alt_description") or ""
            desc = img.get("description") or ""
            tags_text = " ".join([t["title"] for t in img.get("tags", []) if "title" in t])
            metadata = f"{alt} {desc} {tags_text}"
            url_hd = img["urls"].get("full")
            if not url_hd:
                continue

            if is_valid_metadata(metadata, is_fallback=is_fallback) and not is_duplicate_image(url_hd):
                save_image_to_history(url_hd)
                return url_hd
        return None

    def fetch_pexels_image(search_term: str, is_fallback=False):
        headers = {"Authorization": PEXELS_API_KEY}
        params = {
            "query": f"{search_term} pot indoor",
            "orientation": "portrait",
            "size": "large",
            "per_page": 30
        }
        url = "https://api.pexels.com/v1/search"
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            print(f"Pexels error: {resp.status_code}")
            return None

        photos = resp.json().get("photos", [])
        random.shuffle(photos)
        for photo in photos:
            desc = (photo.get("alt") or "").lower()
            url_hd = photo["src"].get("original")
            if not url_hd:
                continue

            if is_valid_metadata(desc, is_fallback=is_fallback) and not is_duplicate_image(url_hd):
                save_image_to_history(url_hd)
                return url_hd
        return None

    for i, term in enumerate(fallback_terms):
        is_fallback = (i > 0)
        img_url = fetch_unsplash_image(term, is_fallback=is_fallback)
        if img_url:
            print(f"✅ Image found on Unsplash for '{term}'")
            return img_url

        img_url = fetch_pexels_image(term, is_fallback=is_fallback)
        if img_url:
            print(f"✅ Image found on Pexels for '{term}'")
            return img_url

    raise Exception(f"No HD image found for '{query}' or fallback terms via Unsplash or Pexels.")



def is_duplicate_image(url: str, history_file: str = "history.txt") -> bool:
    image_id = extract_image_id(url)
    if not os.path.exists(history_file):
        return False
    with open(history_file, "r") as f:
        return image_id.strip() in (line.strip() for line in f.readlines())

def save_image_to_history(url: str, history_file: str = "history.txt"):
    image_id = extract_image_id(url)
    with open(history_file, "a") as f:
        f.write(image_id.strip() + "\n")


# Helper: limit to 30 hashtags max
def limit_hashtags(caption: str) -> str:
    """
    Truncates hashtags in the caption to the first 30 occurrences,
    but preserves all other whitespace and formatting.
    """
    # Find all hashtags
    tags = re.findall(r"#\w+", caption)
    if len(tags) <= 30:
        return caption

    # Determine which tags to keep
    keep = set(tags[:30])

    # Walk through the caption, building output piece by piece
    out = []
    for part in re.split(r"(#\w+)", caption):
        # If it's a hashtag, decide to keep or drop.
        if part.startswith("#"):
            if part in keep:
                out.append(part)
                # Once we've appended it, remove from 'keep' so duplicates beyond first aren't counted
                keep.remove(part)
            # else: drop it (append nothing)
        else:
            # Non-hashtag text: preserve exactly as-is
            out.append(part)

    # Join everything and then collapse any runs of more than one space into a single space only *between* words,
    # but keep leading/trailing spaces intact.
    result = "".join(out)

    # Now tidy up only excessive *inter-word* spaces:
    # Replace three or more spaces with exactly two (as a visual “buffer”),
    # and two spaces with one, but leave single spaces alone.
    # You can tweak these rules if you want to preserve double-spaces after sentences, etc.
    def collapse(match):
        s = match.group(0)
        if len(s) >= 3:
            return "  "
        else:
            return " "

    return re.sub(r" {2,}", collapse, result)


# List of plants, hook list, and helper functions (search_plant_image, generate_caption, build_query) go here
# ... [Keep your existing search_plant_image and generate_caption implementations]

def publish_photo(image_url: str, caption: str) -> dict:
    """
    Uploads an image to Instagram via Graph API and publishes it.
    Returns the API response as a dict.
    """
    # Enforce hashtag limit
    caption = limit_hashtags(caption)
    print(f"Generated caption after limit: {caption}")
    # 1. Create a media container
    create_endpoint = f"https://graph.facebook.com/v17.0/{IG_USER_ID}/media"
    create_payload = {
        "image_url": image_url,
        "caption": caption,
        "access_token": ACCESS_TOKEN
    }
    resp = requests.post(create_endpoint, data=create_payload)
    if resp.status_code != 200:
        print("❌ create-media error:", resp.json())
        return resp.json()
    container_id = resp.json().get("id")
    print("✅ media container id:", container_id)

    # 2. Publish the container
    publish_endpoint = f"https://graph.facebook.com/v17.0/{IG_USER_ID}/media_publish"
    publish_payload = {
        "creation_id": container_id,
        "access_token": ACCESS_TOKEN
    }
    pub_resp = requests.post(publish_endpoint, data=publish_payload)
    if pub_resp.status_code != 200:
        print("❌ publish-media error:", pub_resp.json())
        return pub_resp.json()
    print("✅ publish success:", pub_resp.json())
    return pub_resp.json()



def build_query(plant_name):
    # We include the plant name and additional keywords for context
    # e.g. "Monstera deliciosa indoor plant pot"
    return f"{plant_name} indoor pot"

# Function: Generate Instagram Caption using Gemini
@tool
def generate_caption(plant: str, image_url: str) -> str:
    """
    Use Gemini to create a caption with emojis and hashtags for the given plant.
    Adds consistent brand hashtags and formatting.
    """
    hook = random.choice(HOOK_LIST)
    prompt = (
        f"Line 1: A catchy hook—embellish this seed hook with relevant emojis: \"{hook}\".\n"
        f"Line 2: A fun fact about \"{plant}\" followed by 3–5 trending plant hashtags.\n"
        f"Keep the entire caption under 1000 characters, with no extra labels or commentary."
    )

    messages = [
        SystemMessage(content="You are a witty Instagram content creator for plant lovers."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    gemini_caption = response.content.strip()

    final_caption = f"""{gemini_caption}

☘️✨🌿 ———————————— 🌿✨☘️
💚 Follow 👉 @blurry_plant_love for rare plant moments, green vibes & daily leafy joy!
📸 Share this post with your favorite plant parent 🌱
🪴 Save for plantspiration!
☘️✨🌿 ———————————— 🌿✨☘️

#plantsmakepeoplehappy #plantsofinstagram #houseplants #indoorplants #urbanjungle #plantlover #plantstagram #plantsplantsplants #houseplantclub #plantsarefriends #houseplantjournal #plantparenthood #greeninterior #livingwithplants #myplantlovinghome #plantladyisthenewcatlady #ihavethisthingwithplants #rareplants #plantshop #plantcollection #plantlove #plantcommunity #greenthumb #botanical #greenery #succulents #cactus #gardening #naturelovers #homedecor
"""

    return final_caption


def save_image_locally(image_image_url: str, plant_name: str, output_dir: str = "output") -> str:
    """
    Downloads the image from the provided URL and saves it locally
    in the specified output directory with the plant name and timestamp as filename.
    Returns the local file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp and format filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{plant_name.replace(' ', '_').lower()}_{timestamp}.jpg"
    output_path = os.path.join(output_dir, file_name)

    try:
        img_data = requests.get(image_url).content
        with open(output_path, "wb") as out_file:
            out_file.write(img_data)
        print(f"Image saved locally at: {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to save image locally for {plant_name}: {e}")
        return ""


def enhance_plant_image(input_path, output_path):
    # Read image using OpenCV
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to HSV to enhance green colors
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Boost saturation for green hues
    h, s, v = cv2.split(hsv)
    mask_green = (h > 35) & (h < 90)  # green hue range
    s[mask_green] = np.clip(s[mask_green] * 1.5, 0, 255).astype(np.uint8)

    hsv_enhanced = cv2.merge([h, s, v])
    enhanced_rgb = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    # Convert to PIL for brightness/contrast/sharpness
    img = Image.fromarray(enhanced_rgb)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    # Save the enhanced image
    img.save(output_path)
    print(f"Enhanced image saved to {output_path}")

def upload_to_cloudinary(local_path: str) -> str:
    result = cloudinary.uploader.upload(local_path)
    return result["secure_url"]

def extract_image_id(url: str) -> str:
    """
    Extracts a unique image ID from Unsplash or Pexels URLs.
    - Unsplash: photo-1519836627919-5b9a1653d03b
    - Pexels:   pexels-photo-7959572
    """
    if "unsplash.com" in url:
        match = re.search(r"(photo-\w+)", url)
    elif "pexels.com" in url:
        match = re.search(r"(pexels-photo-\d+)", url)
    else:
        match = None

    return match.group(1) if match else url.split("/")[-1].split("?")[0]

# Main execution
if __name__ == "__main__":
    # 1. Select a random plant
    plant_name = random.choice(PLANT_LIST)
    plant_query=build_query(plant_name)
    print(f"Selected plant: {plant_name}")
    print(f"Selected plant: {plant_query}")

    # 2. Fetch image URL
    image_url = search_plant_image(plant_query)
    print(f"Fetched image URL: {image_url}")

    # 3. Generate caption
    caption = generate_caption.invoke({"plant": plant_name, "image_url": image_url})
    print(f"Generated caption: {caption}")

    # 4. Save image locally
    local_image_path = save_image_locally(image_url, plant_name)

    # 4.1 Enhance the image before upload
    enhanced_path = local_image_path.replace(".jpg", "_enhanced.jpg")
    enhance_plant_image(local_image_path, enhanced_path)
    print(f"Enhanced image saved at: {enhanced_path}")

    # 5. Upload enhanced image to Cloudinary
    cloudinary_url = upload_to_cloudinary(enhanced_path)

    # 6. Publish using Imgur URL
    result = publish_photo(cloudinary_url, caption)
    print(f"Post published: {result}")

["link"]

