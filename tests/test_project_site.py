"""Smoke checks for the static project page shipped with the code release."""

from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.local_links = []
        self.images = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.local_links.append(attrs["href"])
        if tag == "img":
            self.images.append(attrs)
        if tag == "script":
            self.scripts.append(attrs)


def test_project_site_has_complete_static_navigation():
    page = (SITE / "index.html").read_text(encoding="utf-8")
    parser = PageParser()
    parser.feed(page)

    assert {"top", "physical", "simulation", "method", "code"} <= parser.ids
    assert "VLA-Fabric" in page
    assert "80.5" in page and "94.5" in page
    assert any(script.get("src") == "static/app.js" for script in parser.scripts)
    assert "TBD" not in page


def test_project_site_local_assets_and_anchors_resolve():
    page = (SITE / "index.html").read_text(encoding="utf-8")
    parser = PageParser()
    parser.feed(page)

    for image in parser.images:
        assert image.get("alt", "").strip()
        if "src" in image:
            assert (SITE / image["src"]).is_file(), image["src"]

    for href in parser.local_links:
        if href.startswith(("https://", "http://", "mailto:")):
            continue
        if href.startswith("#"):
            assert href[1:] in parser.ids, href
        else:
            assert (SITE / href).exists(), href

    assert (SITE / ".nojekyll").is_file()
