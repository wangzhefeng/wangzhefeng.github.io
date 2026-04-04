import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check-content-images.py"
SPEC = importlib.util.spec_from_file_location("check_content_images", MODULE_PATH)
cci = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(cci)


class CheckContentImagesTests(unittest.TestCase):
    def test_page_bundle_relative_image_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "content"
            post_dir = root / "post" / "sample-post"
            images_dir = post_dir / "images"
            images_dir.mkdir(parents=True)
            (post_dir / "index.md").write_text("![img](./images/example.png)\n", encoding="utf-8")
            (images_dir / "example.png").write_bytes(b"png")

            missing = cci.find_missing_assets(root)

            self.assertEqual(missing, [])

    def test_case_mismatch_is_reported_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "content"
            post_dir = root / "post" / "sample-post"
            images_dir = post_dir / "images"
            images_dir.mkdir(parents=True)
            (post_dir / "index.md").write_text("![img](./images/example.png)\n", encoding="utf-8")
            (images_dir / "Example.png").write_bytes(b"png")

            missing = cci.find_missing_assets(root)

            self.assertEqual(missing, [(post_dir / "index.md", "./images/example.png")])


if __name__ == "__main__":
    unittest.main()
