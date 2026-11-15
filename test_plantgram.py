import os
import sys
import unittest
from unittest.mock import MagicMock, patch

for _mod in ("cloudinary", "cloudinary.uploader", "cv2", "numpy", "PIL", "PIL.Image", "PIL.ImageEnhance"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

import main as plantgram


class TestPlantgram(unittest.TestCase):
    def test_limit_hashtags_caps_at_30(self):
        tags = " ".join(f"#tag{i}" for i in range(40))
        out = plantgram.limit_hashtags(f"Hello {tags}")
        self.assertEqual(len([w for w in out.split() if w.startswith("#")]), 30)

    def test_extract_image_id_unsplash_and_pexels(self):
        self.assertEqual(
            plantgram.extract_image_id("https://images.unsplash.com/photo-1519836627919?auto=format"),
            "photo-1519836627919",
        )
        self.assertEqual(
            plantgram.extract_image_id("https://images.pexels.com/photos/7959572/pexels-photo-7959572.jpeg"),
            "pexels-photo-7959572",
        )
        self.assertEqual(
            plantgram.extract_image_id("https://cdn.pixabay.com/photo/2020/04/12/123456_1280.jpg"),
            "pixabay-123456",
        )
        self.assertEqual(
            plantgram.extract_image_id("https://inaturalist-open-data.s3.amazonaws.com/photos/987654/large.jpg"),
            "inat-987654",
        )

    def test_score_image_candidate_prefers_4_5_portrait_and_likes(self):
        portrait_popular = {"width": 1080, "height": 1350, "likes": 1200, "description": "sunlight monstera leaf"}
        landscape_plain = {"width": 1920, "height": 1080, "likes": 10, "description": "plant"}
        self.assertGreater(
            plantgram.score_image_candidate(portrait_popular),
            plantgram.score_image_candidate(landscape_plain),
        )

    @patch("main.requests.get")
    def test_fetch_inaturalist_candidates(self, mock_get):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "results": [{
                "faves_count": 5,
                "photos": [{"id": 555111, "url": "https://static.inaturalist.org/photos/555111/medium.jpg"}],
            }]
        }
        mock_get.return_value = mock_resp
        with patch("main.is_duplicate_image", return_value=False):
            cands = plantgram.fetch_inaturalist_candidates("Monstera")
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["history_id"], "inat-555111")
        self.assertIn("large.jpg", cands[0]["url"])

    @patch("main.requests.get")
    def test_ensure_long_lived_ig_token_permanent_page_token(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"is_valid": True, "expires_at": 0}}
        mock_get.return_value = mock_resp

        with patch.object(plantgram, "ACCESS_TOKEN", "perm_tok"):
            tok = plantgram.ensure_long_lived_ig_token()
            self.assertEqual(tok, "perm_tok")

    @patch("main.requests.get")
    def test_ensure_long_lived_ig_token_exchanges_for_permanent_page_token(self, mock_get):
        dbg_resp = MagicMock()
        dbg_resp.json.return_value = {"data": {"is_valid": True, "expires_at": 1000}}

        ex_resp = MagicMock()
        ex_resp.json.return_value = {"access_token": "long_user_tok"}

        pages_resp = MagicMock()
        pages_resp.json.return_value = {
            "data": [{
                "id": "page_1",
                "name": "Plantgram",
                "access_token": "perm_page_tok",
                "instagram_business_account": {"id": "ig_123"},
            }]
        }
        mock_get.side_effect = [dbg_resp, ex_resp, pages_resp]

        with patch.object(plantgram, "ACCESS_TOKEN", "short_tok"), \
             patch.object(plantgram, "IG_USER_ID", "ig_123"), \
             patch.dict(os.environ, {"FB_APP_ID": "app1", "FB_APP_SECRET": "sec1"}), \
             patch("main._persist_env_var"):
            tok = plantgram.ensure_long_lived_ig_token()
            self.assertEqual(tok, "perm_page_tok")


if __name__ == "__main__":
    unittest.main()
