import unittest

from runCrawler import clean_url

class TestURLCanonicalisation(unittest.TestCase):
    """
      Convert the scheme and host to lower case: HTTP://www.Example.com/SomeFile.html → http://www.example.com/SomeFile.html
      Remove port 80 from http URLs, and port 443 from HTTPS URLs: http://www.example.com:80 → http://www.example.com
      Make relative URLs absolute: if you crawl http://www.example.com/a/b.html and find the URL ../c.html, it should canonicalize to http://www.example.com/c.html.
      Remove the fragment, which begins with #: http://www.example.com/a.html#anything → http://www.example.com/a.html
      Remove duplicate slashes: http://www.example.com//a.html → http://www.example.com/a.html
      """

    def test_url_case(self):
        self.assertEqual(clean_url('HTTP://www.Example.com/SomeFile.html'), 'http://www.example.com/SomeFile.html')

    def test_url_port(self):
        self.assertEqual(clean_url('http://www.example.com:80'), 'http://www.example.com')

    def test_url_rel(self):
        self.assertEqual(clean_url('../c.html', host_url="http://www.example.com/a/b.html"), 'http://www.example.com/c.html')

    def test_url_fragment(self):
        self.assertEqual(clean_url('http://www.example.com/a.html#anything'), 'http://www.example.com/a.html')

    def test_url_slash(self):
        self.assertEqual(clean_url('http://www.example.com//a.html'), 'http://www.example.com/a.html')

    def test_url_basic(self):
        self.assertEqual(clean_url('http://www.example.com'), 'http://www.example.com')

    def test_url_basic(self):
        self.assertEqual(clean_url('//tools.wmflabs.org/geohack/geohack.php?pagename=Tilaiya_Dam&params=24_19_26_N_85_31_16_E_type:landmark'), 'http://www.example.com')



if __name__ == '__main__':
    unittest.main()
