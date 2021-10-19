import unittest, secretpy
from data_utils import *

fpath = "data/text/alice.txt"
alphabet_name = "basic"
alphabet = get_alphabet(alphabet_name)


class TestDataUtils(unittest.TestCase):
    def test_df_generation_plain(self):
        """Test if the DataFrame is generated as expected when calling gen_df_plain"""
        txt = load_txt(fpath, alphabet_name, 0)
        df = gen_df_plain(txt)

        for t, e in zip(df.Txt, df.Enc):
            self.assertEqual(t, e)

        for k in df.Key:
            self.assertEqual(k, "")

        for l in df.Label:
            self.assertEqual(l, "plain")

        for c in df.Cipher:
            self.assertEqual(c, "plain")

    def test_df_generation_cipher(self):
        """Test if the DataFrame is generated as expected when calling gen_df_cipher"""
        txt = load_txt(fpath, alphabet_name, 0)

        df = _gen_df_cipher(txt, alphabet_name)

        df_caesar = df[df.Cipher == "caesar"]

        for t, e, k in zip(df_caesar.Txt, df_caesar.Enc, df_caesar.Key):
            e_ = secretpy.Caesar().encrypt(t, k, alphabet)
            self.assertEqual(e_, e)

        for c in df_caesar.Cipher:
            self.assertEqual(c, "caesar")

    def test_encode_txt(self):
        txt = load_txt(fpath, alphabet_name, 0)
        one_hot = encode_txt(txt, alphabet_name)

        for sample in one_hot:
            n_ones = sample.sum(dim=0)
            self.assertEqual(n_ones.max(), 1)

    def test_gen_df(self):
        """Test if the DataFrame is generated as expected when calling gen_df(fpath)"""
        df = _gen_df(fpath, alphabet_name)
        len_cipher = df.groupby("Cipher").size().reset_index(name="Count").Count
        len_cipher[len_cipher[0] == len_cipher].all()


if __name__ == "__main__":
    unittest.main()
