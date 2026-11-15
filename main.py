import os
import re
import time
import random
import requests
from datetime import datetime
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://graph.facebook.com/v19.0"
HTTP_TIMEOUT = 60

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Instagram Graph API credentials
ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN")  # Permanent Page token (or long-lived token)
IG_USER_ID = os.getenv("IG_USER_ID")        # Your Instagram Business Account ID


class SystemMessage:
    def __init__(self, content: str):
        self.content = content


class HumanMessage:
    def __init__(self, content: str):
        self.content = content


class _GeminiResponse:
    def __init__(self, content: str):
        self.content = content


class ChatGoogleGenerativeAI:
    def __init__(self, model: str = "models/gemini-flash-lite-latest", temperature: float = 0.7):
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.temperature = temperature

    def invoke(self, messages: list) -> _GeminiResponse:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        sys_parts = [m.content for m in messages if isinstance(m, SystemMessage)]
        user_parts = [m.content for m in messages if not isinstance(m, SystemMessage)]
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "\n\n".join(user_parts)}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if sys_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(sys_parts)}]}
        for candidate_model in (self.model, "models/gemini-3.1-flash-lite", "models/gemini-flash-latest"):
            url = f"https://generativelanguage.googleapis.com/v1beta/{candidate_model}:generateContent?key={api_key}"
            resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return _GeminiResponse(text)
        resp.raise_for_status()


# Initialize Gemini Chat Model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-lite-latest",
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


BANNED_IMAGE_KEYWORDS = {
    "illustration", "vector", "drawing", "artificial", "render", "3d", "toy",
    "fake", "vessel", "cartoon", "clipart", "logo", "text", "watermark",
}
PLANT_KEYWORDS = {
    "plant", "leaf", "leaves", "foliage", "greenery", "houseplant",
    "indoor", "pot", "potted", "planter", "botanical", "succulent", "fern", "cactus",
}
AESTHETIC_BONUS_KEYWORDS = {
    "sunlight", "window", "minimalist", "aesthetic", "monstera", "living room",
    "ceramic", "terracotta", "morning light", "close up", "macro",
}


def _is_valid_plant_metadata(text: str) -> bool:
    lower = (text or "").lower()
    if any(bad in lower for bad in BANNED_IMAGE_KEYWORDS):
        return False
    if not lower.strip():
        return True
    return any(good in lower for good in PLANT_KEYWORDS)


def score_image_candidate(cand: dict) -> float:
    """
    Fast, zero-download heuristic score (0..100+) prioritizing:
    1. Instagram 4:5 portrait aspect ratio (height/width ~ 1.25)
    2. High resolution (>= 1080px wide)
    3. Community likes/downloads (Unsplash/Pixabay social proof)
    4. Botanical/aesthetic keyword relevance
    """
    w = cand.get("width") or 1000
    h = cand.get("height") or 1250
    ratio = h / max(w, 1)

    # Best score near 1.25 (4:5 portrait); penalize landscape (< 0.95)
    if ratio < 0.95:
        aspect_score = -25.0
    else:
        aspect_score = max(0.0, 35.0 - abs(ratio - 1.25) * 30.0)

    res_score = min(w / 1080.0, 1.5) * 20.0
    likes = cand.get("likes") or 0
    popularity_score = min(likes, 4000) / 80.0  # up to +50 pts

    desc = (cand.get("description") or "").lower()
    aesthetic_hits = sum(1 for kw in AESTHETIC_BONUS_KEYWORDS if kw in desc)
    keyword_score = min(aesthetic_hits * 4.0, 16.0)

    return aspect_score + res_score + popularity_score + keyword_score


def fetch_unsplash_candidates(search_term: str, per_page: int = 30) -> list:
    key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not key:
        return []
    params = {
        "query": search_term,
        "client_id": key,
        "per_page": per_page,
        "orientation": "portrait",
        "content_filter": "high",
    }
    try:
        resp = requests.get("https://api.unsplash.com/search/photos", params=params, timeout=15)
        if resp.status_code != 200:
            return []
        out = []
        for img in resp.json().get("results", []):
            url_hd = (img.get("urls") or {}).get("regular") or (img.get("urls") or {}).get("full")
            if not url_hd or is_duplicate_image(url_hd):
                continue
            alt = img.get("alt_description") or ""
            desc = img.get("description") or ""
            tags_text = " ".join(t["title"] for t in img.get("tags", []) if isinstance(t, dict) and "title" in t)
            meta = f"{alt} {desc} {tags_text}"
            if not _is_valid_plant_metadata(meta):
                continue
            out.append({
                "url": url_hd,
                "width": img.get("width") or 1080,
                "height": img.get("height") or 1350,
                "likes": img.get("likes") or 0,
                "description": meta,
                "source": "unsplash",
            })
        return out
    except Exception as e:
        print(f"⚠️ Unsplash error: {e}")
        return []


def fetch_pexels_candidates(search_term: str, per_page: int = 30) -> list:
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        return []
    headers = {"Authorization": key}
    params = {"query": search_term, "orientation": "portrait", "size": "large", "per_page": per_page}
    try:
        resp = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            return []
        out = []
        for photo in resp.json().get("photos", []):
            src = photo.get("src") or {}
            url_hd = src.get("large2x") or src.get("original")
            if not url_hd or is_duplicate_image(url_hd):
                continue
            desc = photo.get("alt") or ""
            if not _is_valid_plant_metadata(desc):
                continue
            out.append({
                "url": url_hd,
                "width": photo.get("width") or 1080,
                "height": photo.get("height") or 1350,
                "likes": 40,
                "description": desc,
                "source": "pexels",
            })
        return out
    except Exception as e:
        print(f"⚠️ Pexels error: {e}")
        return []


def fetch_pixabay_candidates(search_term: str, per_page: int = 30) -> list:
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        return []
    params = {
        "key": key,
        "q": search_term,
        "image_type": "photo",
        "orientation": "vertical",
        "safesearch": "true",
        "per_page": per_page,
    }
    try:
        resp = requests.get("https://pixabay.com/api/", params=params, timeout=15)
        if resp.status_code != 200:
            return []
        out = []
        for hit in resp.json().get("hits", []):
            url_hd = hit.get("largeImageURL") or hit.get("fullHDURL")
            if not url_hd:
                continue
            pid = f"pixabay-{hit.get('id')}"
            if is_duplicate_image(pid):
                continue
            tags = hit.get("tags") or ""
            if not _is_valid_plant_metadata(tags):
                continue
            likes = (hit.get("likes") or 0) + (hit.get("downloads") or 0) // 25
            out.append({
                "url": url_hd,
                "history_id": pid,
                "width": hit.get("imageWidth") or 1080,
                "height": hit.get("imageHeight") or 1350,
                "likes": likes,
                "description": tags,
                "source": "pixabay",
            })
        return out
    except Exception as e:
        print(f"⚠️ Pixabay error: {e}")
        return []


def fetch_inaturalist_candidates(plant_name: str, per_page: int = 12) -> list:
    """
    Free CC0/CC-BY research-grade botanical photography from iNaturalist API (no key required).
    Great for rare plants not found in stock libraries.
    """
    params = {
        "q": plant_name,
        "quality_grade": "research",
        "photos": "true",
        "photo_license": "cc0,cc-by",
        "per_page": per_page,
    }
    try:
        resp = requests.get("https://api.inaturalist.org/v1/observations", params=params, timeout=15)
        if resp.status_code != 200:
            return []
        out = []
        for obs in resp.json().get("results", []):
            for photo in obs.get("photos", [])[:1]:
                raw_url = photo.get("url") or ""
                if not raw_url:
                    continue
                large_url = raw_url.replace("square.", "large.").replace("medium.", "large.")
                pid = f"inat-{photo.get('id')}"
                if is_duplicate_image(pid):
                    continue
                dims = photo.get("original_dimensions") or {}
                out.append({
                    "url": large_url,
                    "history_id": pid,
                    "width": dims.get("width") or 1024,
                    "height": dims.get("height") or 1280,
                    "likes": (obs.get("faves_count") or 0) * 15,
                    "description": f"{plant_name} botanical leaf plant",
                    "source": "inaturalist",
                })
        return out
    except Exception as e:
        print(f"⚠️ iNaturalist error: {e}")
        return []


def check_image_aesthetics(image_path: str) -> bool:
    """
    Fast (<0.15s) OpenCV quality gate: rejects blurry, tiny, or poorly exposed photos.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    if min(w, h) < 700:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    if mean_luma < 35.0 or mean_luma > 235.0:
        return False
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return sharpness >= 35.0


def select_best_plant_image(plant_name: str, output_dir: str = "output") -> tuple:
    """
    Gathers free plant photos from Unsplash, Pexels, Pixabay (and iNaturalist fallback),
    ranks them in <1s by 4:5 portrait fit, resolution, and popularity, and verifies the top
    candidate with a fast OpenCV sharpness/exposure check before recording it in history.txt.
    Returns (selected_url, local_path, source_name).
    """
    queries = [
        f"{plant_name} aesthetic houseplant",
        f"{plant_name} indoor plant",
        f"sunlit houseplant indoor pot",
    ]
    all_candidates = []
    seen_ids = set()

    for q_idx, q in enumerate(queries):
        batch = []
        batch.extend(fetch_unsplash_candidates(q, per_page=20))
        batch.extend(fetch_pexels_candidates(q, per_page=20))
        batch.extend(fetch_pixabay_candidates(q, per_page=20))
        if q_idx == 0 and len(batch) < 4:
            batch.extend(fetch_inaturalist_candidates(plant_name, per_page=10))

        for c in batch:
            cid = c.get("history_id") or extract_image_id(c["url"])
            if cid not in seen_ids and not is_duplicate_image(cid):
                seen_ids.add(cid)
                c["history_id"] = cid
                c["score"] = score_image_candidate(c)
                all_candidates.append(c)

        if len(all_candidates) >= 12:
            break

    if not all_candidates:
        raise RuntimeError(f"No unique HD plant image found for '{plant_name}'.")

    all_candidates.sort(key=lambda c: c["score"], reverse=True)
    print(f"🔍 Ranked {len(all_candidates)} unique candidates for '{plant_name}' (top score: {all_candidates[0]['score']:.1f} from {all_candidates[0]['source']})")

    for cand in all_candidates[:3]:
        local_path = save_image_locally(cand["url"], plant_name, output_dir=output_dir)
        if not local_path:
            continue
        if check_image_aesthetics(local_path):
            save_image_to_history(cand["history_id"])
            print(f"✅ Selected best image ({cand['source']}, score={cand['score']:.1f}): {cand['url']}")
            return cand["url"], local_path, cand["source"]
        print(f"⚠️ Candidate failed fast quality gate, trying next...")

    # Fallback: use top downloaded candidate if all 3 failed strict threshold
    fallback = all_candidates[0]
    local_path = save_image_locally(fallback["url"], plant_name, output_dir=output_dir)
    if not local_path:
        raise RuntimeError(f"Failed to download selected image for '{plant_name}'.")
    save_image_to_history(fallback["history_id"])
    return fallback["url"], local_path, fallback["source"]


def search_plant_image(query: str) -> str:
    """Backwards-compatible helper returning the best image URL."""
    url, _, _ = select_best_plant_image(query)
    return url


def is_duplicate_image(url_or_id: str, history_file: str = "history.txt") -> bool:
    image_id = extract_image_id(url_or_id) if "://" in url_or_id else url_or_id.strip()
    history_path = history_file if os.path.isabs(history_file) else os.path.join(BASE_DIR, history_file)
    if not os.path.exists(history_path):
        return False
    with open(history_path, "r", encoding="utf-8") as f:
        return image_id.strip() in (line.strip() for line in f.readlines())


def save_image_to_history(url_or_id: str, history_file: str = "history.txt"):
    image_id = extract_image_id(url_or_id) if "://" in url_or_id else url_or_id.strip()
    history_path = history_file if os.path.isabs(history_file) else os.path.join(BASE_DIR, history_file)
    with open(history_path, "a", encoding="utf-8") as f:
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

def _persist_env_var(key: str, value: str) -> None:
    """Updates a key=value entry in the local .env file if it exists."""
    env_file = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_file):
        return
    with open(env_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}")
    with open(env_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def ensure_long_lived_ig_token() -> str:
    """
    Checks IG_ACCESS_TOKEN expiration via /debug_token.
    - If expires_at == 0, it is already a permanent non-expiring Page token.
    - If < 45 days remain (or it is a User token) and FB_APP_ID/FB_APP_SECRET are set,
      automatically exchanges for a 60-day User token and derives a permanent
      non-expiring Page Access Token (expires_at == 0) via /me/accounts.
    """
    global ACCESS_TOKEN
    if not ACCESS_TOKEN:
        return ""

    fb_app_id = os.getenv("FB_APP_ID", "")
    fb_app_secret = os.getenv("FB_APP_SECRET", "")
    app_token = f"{fb_app_id}|{fb_app_secret}" if (fb_app_id and fb_app_secret) else ACCESS_TOKEN

    try:
        dbg_resp = requests.get(
            f"{BASE_URL}/debug_token",
            params={"input_token": ACCESS_TOKEN, "access_token": app_token},
            timeout=HTTP_TIMEOUT,
        ).json()
        data = dbg_resp.get("data", {})
        is_valid = data.get("is_valid", False)
        expires_at = data.get("expires_at")

        if not is_valid:
            print(f"⚠️ Instagram token is invalid/expired: {data.get('error', dbg_resp)}")
            return ACCESS_TOKEN

        if expires_at == 0:
            print("🔒 Instagram token is a permanent non-expiring Page token.")
            return ACCESS_TOKEN

        remaining_days = (expires_at - time.time()) / 86400 if expires_at else 0
        print(f"🔑 Instagram token expires in {remaining_days:.1f} days.")

        if remaining_days < 45 and fb_app_id and fb_app_secret:
            print("🔄 Exchanging/refreshing Instagram token for a permanent Page token...")
            source_token = os.getenv("IG_USER_TOKEN") or ACCESS_TOKEN
            ex_resp = requests.get(
                f"{BASE_URL}/oauth/access_token",
                params={
                    "grant_type": "fb_exchange_token",
                    "client_id": fb_app_id,
                    "client_secret": fb_app_secret,
                    "fb_exchange_token": source_token,
                },
                timeout=HTTP_TIMEOUT,
            ).json()
            long_user_token = ex_resp.get("access_token")
            if long_user_token:
                _persist_env_var("IG_USER_TOKEN", long_user_token)
                os.environ["IG_USER_TOKEN"] = long_user_token
                final_token = long_user_token

                pages_resp = requests.get(
                    f"{BASE_URL}/me/accounts",
                    params={
                        "fields": "id,name,access_token,instagram_business_account",
                        "access_token": long_user_token,
                    },
                    timeout=HTTP_TIMEOUT,
                ).json()
                for page in pages_resp.get("data", []):
                    ig_acct = (page.get("instagram_business_account") or {}).get("id")
                    if ig_acct == IG_USER_ID and page.get("access_token"):
                        final_token = page["access_token"]
                        print("✅ Obtained permanent non-expiring Page/Instagram access token via /me/accounts!")
                        break

                ACCESS_TOKEN = final_token
                os.environ["IG_ACCESS_TOKEN"] = final_token
                _persist_env_var("IG_ACCESS_TOKEN", final_token)
                print("✅ Refreshed IG_ACCESS_TOKEN saved to .env.")
    except Exception as e:
        print(f"⚠️ Instagram token check/refresh warning: {e}")

    return ACCESS_TOKEN


def wait_for_media_container(container_id: str, max_wait_seconds: int = 120, poll_interval: int = 3) -> None:
    """
    Polls the Instagram Graph API container status until it is FINISHED processing.
    Prevents Graph API error 9007 ('Media ID is not available') on publish.
    """
    endpoint = f"{BASE_URL}/{container_id}"
    params = {"fields": "status_code,status", "access_token": ACCESS_TOKEN}
    elapsed = 0
    while elapsed < max_wait_seconds:
        resp = requests.get(endpoint, params=params, timeout=HTTP_TIMEOUT).json()
        status_code = resp.get("status_code")
        print(f"⏳ Instagram container {container_id} status: {status_code or resp}")
        if status_code == "FINISHED":
            return
        if status_code in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"Instagram media container processing failed: {resp}")
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"Timed out waiting for Instagram container {container_id} to finish processing.")


def publish_photo(image_url: str, caption: str) -> dict:
    """
    Uploads an image to Instagram via Graph API v19.0 and publishes it.
    Returns the API response as a dict.
    """
    ensure_long_lived_ig_token()
    caption = limit_hashtags(caption)
    print(f"Generated caption after limit: {caption}")

    # 1. Create a media container
    create_endpoint = f"{BASE_URL}/{IG_USER_ID}/media"
    create_payload = {
        "image_url": image_url,
        "caption": caption,
        "access_token": ACCESS_TOKEN
    }
    resp = requests.post(create_endpoint, data=create_payload, timeout=HTTP_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"❌ create-media error: {resp.json()}")
    container_id = resp.json().get("id")
    print("✅ media container id:", container_id)

    # 2. Wait for container to finish processing before publishing
    wait_for_media_container(container_id)

    # 3. Publish the container
    publish_endpoint = f"{BASE_URL}/{IG_USER_ID}/media_publish"
    publish_payload = {
        "creation_id": container_id,
        "access_token": ACCESS_TOKEN
    }
    pub_resp = requests.post(publish_endpoint, data=publish_payload, timeout=HTTP_TIMEOUT)
    if pub_resp.status_code != 200:
        raise RuntimeError(f"❌ publish-media error: {pub_resp.json()}")
    print("✅ publish success:", pub_resp.json())
    return pub_resp.json()


def build_query(plant_name):
    return f"{plant_name} indoor pot"


def generate_caption(plant: str, image_url: str) -> str:
    """
    Generates an engagement-optimized Instagram caption designed for saves, comments,
    and follower growth (Hook -> Save-Worthy Care Tip -> Comment Question -> Clean Hashtags).
    """
    prompt = f"""
You are a top botanical Instagram growth strategist for @blurry_plant_love (a fast-growing houseplant community).

Write a high-engagement caption for a stunning photo of "{plant}" with this exact structure:
1. HOOK LINE: One short, intriguing, relatable or surprising sentence about {plant} (with 1-2 fitting emojis).
2. CARE SECRET (SAVE-WORTHY): A blank line, then "🌱 Care Secret:" followed by 2 crisp, practical sentences on how to make {plant} thrive (light, watering, humidity, or propagation tip people want to save).
3. COMMENT PROMPT: A blank line, then one easy-to-answer question asking readers about their experience with {plant} (e.g., "Do you have one in your collection, or is it still on your wishlist? 👇").

Rules:
- Plain text only (no markdown asterisks **, no HTML).
- Keep the body under 700 characters so it is punchy and easy to read.
- Do NOT include hashtags (I will append curated hashtags separately).
    """

    messages = [
        SystemMessage(content="You write viral, save-worthy Instagram captions for houseplant lovers."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    gemini_caption = re.sub(r"[*_`]", "", response.content.strip())

    plant_tag = "#" + re.sub(r"[^a-zA-Z0-9]", "", plant).lower()
    curated_tags = " ".join(dict.fromkeys([
        plant_tag,
        "#houseplants", "#plantcare", "#urbanjungle",
        "#indoorplants", "#plantlover", "#plantcommunity", "#botanical",
    ]))

    final_caption = (
        f"{gemini_caption}\n\n"
        f"🪴 Save this tip for later & follow @blurry_plant_love for daily plant care + green inspiration.\n\n"
        f"{curated_tags}"
    )
    return final_caption


def save_image_locally(image_image_url: str, plant_name: str, output_dir: str = "output") -> str:
    """
    Downloads the image from the provided URL and saves it locally
    in the specified output directory with the plant name and timestamp as filename.
    Returns the local file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{plant_name.replace(' ', '_').lower()}_{timestamp}.jpg"
    output_path = os.path.join(output_dir, file_name)

    try:
        headers = {"User-Agent": "PlantgramBot/1.0"}
        resp = requests.get(image_image_url, headers=headers, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        with open(output_path, "wb") as out_file:
            out_file.write(resp.content)
        print(f"Image saved locally at: {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to save image locally for {plant_name}: {e}")
        return ""


def enhance_plant_image(input_path: str, output_path: str) -> None:
    """
    Frames the photo to Instagram's optimal 4:5 portrait (1080x1350) for maximum feed
    real estate, and applies subtle, natural botanical color grading.
    """
    image = cv2.imread(input_path)
    if image is None:
        raise RuntimeError(f"Could not read image at {input_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Subtle natural green vibrancy (avoids artificial neon look)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask_green = (h > 35) & (h < 90)
    s[mask_green] = np.clip(s[mask_green] * 1.18, 0, 255).astype(np.uint8)
    enhanced_rgb = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)

    img = Image.fromarray(enhanced_rgb)

    # Center-crop to 4:5 portrait (0.8 aspect ratio) and resize to 1080x1350
    w, h_px = img.size
    target_ratio = 4 / 5
    current_ratio = w / max(h_px, 1)
    if current_ratio > target_ratio:
        new_w = int(h_px * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h_px))
    elif current_ratio < target_ratio:
        new_h = int(w / target_ratio)
        top = (h_px - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    img = img.resize((1080, 1350), Image.LANCZOS)
    img = ImageEnhance.Brightness(img).enhance(1.04)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = ImageEnhance.Sharpness(img).enhance(1.25)

    img.save(output_path, quality=95)
    print(f"Enhanced 4:5 image saved to {output_path}")


def upload_to_cloudinary(local_path: str) -> str:
    result = cloudinary.uploader.upload(local_path)
    return result["secure_url"]


def extract_image_id(url: str) -> str:
    """
    Extracts a unique image ID from Unsplash, Pexels, Pixabay, or iNaturalist URLs.
    """
    if "unsplash.com" in url:
        match = re.search(r"(photo-\w+)", url)
    elif "pexels.com" in url:
        match = re.search(r"(pexels-photo-\d+)", url)
    elif "pixabay.com" in url:
        match = re.search(r"(\d+)_\d+", url)
        if match:
            return f"pixabay-{match.group(1)}"
    elif "inaturalist" in url:
        match = re.search(r"/photos/(\d+)/", url)
        if match:
            return f"inat-{match.group(1)}"
    else:
        match = None

    return match.group(1) if match else url.split("/")[-1].split("?")[0]


# Main execution
if __name__ == "__main__":
    plant_name = random.choice(PLANT_LIST)
    print(f"Selected plant: {plant_name}")

    image_url, local_image_path, source_name = select_best_plant_image(plant_name)
    print(f"Best image ({source_name}): {image_url}")

    caption = generate_caption(plant_name, image_url)
    print(f"Generated caption:\n{caption}")

    enhanced_path = local_image_path.replace(".jpg", "_enhanced.jpg")
    enhance_plant_image(local_image_path, enhanced_path)

    cloudinary_url = upload_to_cloudinary(enhanced_path)

    result = publish_photo(cloudinary_url, caption)
    print(f"Post published: {result}")

