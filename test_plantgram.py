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
