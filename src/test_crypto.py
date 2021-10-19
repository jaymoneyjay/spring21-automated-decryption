import unittest, secretpy


from cipher import *
from data_utils import load_msgs, get_char_order
import alphabet as alph
import random

alphabet_ours = alph.basic()
alphabet_theirs = alph.basic()[1:]
msgs = load_msgs("alice", alphabet_ours, 40, 150)
n_samples = 1
order = get_char_order(msgs, alphabet_ours)

caesar_ours = Caesar(alphabet_ours)
substitution_ours = Substitution(alphabet_ours)
caesar_theirs = secretpy.SaveSpaces(secretpy.CryptMachine(secretpy.Caesar()))
substitution_theirs = secretpy.SaveSpaces(
    secretpy.CryptMachine(secretpy.SimpleSubstitution())
)


class MyTestCase(unittest.TestCase):
    def test_encryption_caesar(self):
        samples = random.sample(msgs, n_samples)
        encs, keys = caesar_ours.encrypt_all(samples, order)

        for (enc, k, msg) in zip(encs, keys, samples):
            caesar_theirs.set_key(k)
            caesar_theirs.set_alphabet(alphabet_theirs)
            enc_ = caesar_theirs.encrypt(msg)
            self.assertEqual(
                enc,
                enc_,
                f"Error when encrypting message {msg} with key {k} and cipher {caesar_ours.id}. Got {enc}, expected {enc_}",
            )

    def test_decryption_caesar(self):
        samples = random.sample(msgs, n_samples)
        encs, keys = caesar_ours.encrypt_all(samples, order)

        for (enc, k, msg) in zip(encs, keys, samples):
            msg_ = caesar_ours.decrypt(enc, k)
            self.assertEqual(
                msg,
                msg_,
                f"Error when decrypting cipher {enc} with key {k} and cipher {caesar_ours.id}. Got {msg}, expected {msg_}",
            )

    def test_encryption_substitution(self):
        samples = random.sample(msgs, n_samples)
        encs, keys = substitution_ours.encrypt_all(samples, order)

        for (enc, k, msg) in zip(encs, keys, samples):
            substitution_theirs.set_key(k)
            substitution_theirs.set_alphabet(alphabet_ours)
            enc_ = substitution_theirs.encrypt(msg)
            self.assertEqual(
                enc,
                enc_,
                f"Error when encrypting message {msg} with key {k} and cipher {substitution_ours.id}. Got {enc}, expected {enc_}",
            )

    def test_decryption_substitution(self):
        samples = random.sample(msgs, n_samples)
        encs, keys = substitution_ours.encrypt_all(samples, order)

        for (enc, k, msg) in zip(encs, keys, samples):
            msg_ = substitution_ours.decrypt(enc, k)
            self.assertEqual(
                msg,
                msg_,
                f"Error when decrypting cipher {enc} with key {k} and cipher {substitution_ours.id}. Got {msg}, expected {msg_}",
            )


if __name__ == "__main__":
    unittest.main()
