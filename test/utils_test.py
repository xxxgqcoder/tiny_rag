import unittest
import os


class TestUtils(unittest.TestCase):

    def test_estimate_token_num(self):
        from utils import estimate_token_num
        text = "中文http://url/key=123 中text 1 123 word"
        want = [
            "中", "文", "http://url/key=123", "中", "text", "1", "123", "word"
        ]
        _, tokens = estimate_token_num(text=text)
        self.assertEqual(tokens, want)
        pass


if __name__ == '__main__':

    unittest.main()
