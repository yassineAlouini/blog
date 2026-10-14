"""Navigation checks: python3 -m unittest discover -s tests."""
import unittest
from html.parser import HTMLParser
from pathlib import Path

from build import page_shell

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.links = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.append(dict(attrs))


class NavigationTests(unittest.TestCase):
    def assert_site_links(self, markup):
        self.assertIn('/assets/site-navigation.css?v=1', markup)
        self.assertIn('assets/style.css?v=shared-navigation-2', markup)
        links = Links(markup).links
        hrefs = {a.get('href') for a in links}
        self.assertTrue({'/', '/blog/', '/sketches/'}.issubset(hrefs))
        home = next(a for a in links if a.get('class') == 'site-home')
        self.assertEqual(home['href'], '/')

    def test_generator_preserves_site_navigation_on_index_and_posts(self):
        for rel, current in [('', 'page'), ('../', 'location')]:
            with self.subTest(rel=rel):
                markup = page_shell('Test', '', '<main id="main-content"></main>', rel=rel)
                self.assert_site_links(markup)
                blog = next(a for a in Links(markup).links if a.get('href') == '/blog/')
                self.assertEqual(blog['aria-current'], current)

    def test_published_pages_have_home_and_cross_section_links(self):
        pages = [ROOT / 'index.html', *sorted((ROOT / 'posts').glob('*.html'))]
        self.assertGreater(len(pages), 1)
        for page in pages:
            with self.subTest(page=page.name):
                markup = page.read_text()
                self.assert_site_links(markup)
                if page.parent.name == 'posts':
                    back = next(a for a in Links(markup).links if a.get('class') == 'back')
                    self.assertEqual(back['href'], '/')
                    self.assertIn('← Home', markup)


if __name__ == '__main__':
    unittest.main()
