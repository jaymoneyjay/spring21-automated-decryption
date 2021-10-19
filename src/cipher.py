""" Classes to provide functionality to encrypt and decrypt messages under specific ciphers. """

import secretpy, base64

from numpy.random import shuffle, randint
from Crypto.Cipher import AES
import random, copy


class SimpleEncoder:
    def __init__(self, encoding=None):
        if encoding is None or encoding == "":
            self.encode = self.decode = lambda x: x
        if encoding == "b64":
            self.encode = base64.b64encode
            self.decode = base64.b64decode
        if encoding == "b85":
            self.encode = base64.b85encode
            self.decode = base64.b85decode


class Cipher:
    def __init__(self, alphabet, cipher_id, save_spaces):
        """Initialize the cipher machines and alphabet.

        Args:
            alphabet (string): The alphabet to use.

            cipher_id (string): ID to specify the type of cipher.
                Values=['plain', 'caesar', 'substitution', 'vigenere', 'AES', 'fakeAES']

            save_spaces (bool): Whether or not to save spaces during the encryption.
                True: spaces in plaintext are spaces in ciphertext.
                False: spaces in plaintext are encrypted as any other character in ciphertext.
        """

        self.alphabet = alphabet
        self.id = cipher_id
        self.cm = self._init_cm(save_spaces)

    def gen_data(
        self,
        msgs,
        order,
        key_rot=1,
        key_gen_strategy="random",
        partial_size=0,
        low_dist_data=False,
    ):
        """ Generate data for the class label incorrect decryption """
        return self.encrypt_all(
            msgs,
            order,
            key_rot=key_rot,
            key_gen_strategy=key_gen_strategy,
            partial_size=partial_size,
        )

    def encrypt_all(
        self, msgs, order, key_rot=1, key_gen_strategy="random", partial_size=0
    ):
        """Encrypt all messages with random keys and the specified cipher.

        Args:
            msgs (list): Input messages to encrypt

            order (string): Order in which to apply partial encryption to chars

            key_rot (int, optional): Frequency of key rotation.
                key_rot=1 -> Key is rotated every message.
                key_rot=50 -> Key is rotated after 50 messages.
                key_rot=0 -> Key is not rotated at all.

            key_gen_strategy (string, optional): ID to specify the key generation algorithm
                Values=['iterative', 'random']

            partial_size (int): The size of the encrypted part of the messages.
                All other characters are masked with a special character.
                partial_size=0 -> All characters are encrypted
                partial_size=4 -> The 4 most common characters are encrypted, the others are masked.

        Returns:
            enc_store (list): List of encrypted messages
            key_store: (list): List of keys used
        """

        enc_store = []
        key_store = []

        if key_rot == 0:
            # Key is not rotated
            n_keys = 1
            key_rot = len(msgs)
        else:
            n_keys = len(msgs) // key_rot + 1

        keys = self.gen_keys(
            n_keys, strategy=key_gen_strategy, order=order, partial_size=partial_size
        )

        for i, m in enumerate(msgs):
            k = keys[i // key_rot]

            enc = self.encrypt(m, k)
            enc_store.append(enc)
            key_store.append(k)

        return enc_store, key_store

    def decrypt_all(self, msgs, keys):
        """Decrypt messages with the specified keys

        Args:
            msgs (list): Input messages to decrypt

            keys (list): Keys used for decryption
        """

        assert len(msgs) == len(
            keys
        ), f"Messages ({len(msgs)}) and keys({len(keys)} are expected to have the same length."
        dec_store = []
        for m, k in zip(msgs, keys):
            dec = self.decrypt(m, k)
            dec_store.append(dec)
        return dec_store

    def encrypt(self, msg, key):
        """Encrypt a single message with the specified key.

        Args:
            msg (string): Input text to encrypt.

            key (string, int): Key to use for encryption.
        """

        self.cm.set_key(key)
        enc = self.cm.encrypt(msg)
        return enc

    def decrypt(self, msg, key):
        """Decrypt a given text with specified cipher and key

        Args:
            msg (string): Input text to decrypt.

            key (string, int): Key to use for decryption.
        """

        self.cm.set_key(key)
        dec = self.cm.decrypt(msg)
        return dec

    def _init_cm(self, save_spaces):
        """Initialize the cipher machine object of secretpy used for all crypto operations."""
        raise NotImplementedError()

    def set_alphabet(self, alphabet):
        """Set the alphabet of the cipher object."""
        self.alphabet = alphabet

    def _gen_key(self, strategy, order, partial_size, prefix):
        """Generate a single key according to the specified parameters."""
        raise NotImplementedError

    def gen_keys(self, n_keys, strategy, order, partial_size=0, prefix=""):
        """Generate a list of keys.

        Args:
            n_keys (int): Number of keys to generate.
                if strategy='iterative' n_keys does not matter as all possible keys are generated.

            strategy (string, optional): ID to specify the key generation algorithm
                Values=['iterative', 'random']

            order (string): The order in which characters are chosen to be encrypted.
                Usually alphabet sorted in decreasing frequencies.

            partial_size (int): The size of the encrypted part of the messages.
                All other characters are masked with a special character.
                partial_size=0 -> All characters are encrypted
                partial_size=4 -> The 4 most common characters are encrypted, the others are masked.
            prefix (string, optional): prefix to use as basis to generate new keys
        """

        keys = []

        if partial_size == 0:
            partial_size = len(self.alphabet) - 1

        for _ in range(n_keys):
            key = self._gen_key(strategy, order, partial_size, prefix)
            keys.append(key)
        return keys


class FakeAes(Cipher):
    """
    Fake Aes Cipher that outputs a random sequence of strings from the alphabet of the same length as the input message.
    """

    def __init__(self, alphabet, save_spaces=True, encoding=None):
        super().__init__(alphabet, "fakeAes", save_spaces)
        self.encoder = SimpleEncoder(encoding)

    def encrypt(self, msg, key):
        """ Encrypt a message under the FakeAes cipher """
        n = len(msg)
        cipher = list(self.alphabet)
        shuffle(cipher)
        cipher = "".join(cipher[:n])

        return cipher

    def _gen_key(self, strategy, order, partial_size, prefix):
        """ Generate an empty key as FakeAes does not require a key """
        return ""

    def _init_cm(self, save_spaces):
        """ Initialize the cipher machine for FakeAes. Returns None as FakeAes does not require a cipher machine """
        return None


class RealAes(Cipher):
    def __init__(self, alphabet, save_spaces=True, encoding=None, iv=None):
        super().__init__(alphabet, "realAes", save_spaces)

        self.__iv = iv or b"5432109876543210"
        self.encoder = SimpleEncoder(encoding)

    def encrypt(self, msg, key):
        """ Encrypt a message under the RealAes cipher """
        if isinstance(msg, str):
            msg = msg.encode("utf-8")

        cipher = AES.new(key, AES.MODE_CBC, self.__iv)
        ctx = cipher.encrypt(msg)
        ctx = self.encoder.encode(ctx)
        return ctx

    def decrypt(self, msg, key):
        """ Decrypt a message under the Real Aes cipher """
        ctx = self.encoder.decode(msg)
        cipher = AES.new(key, AES.MODE_CBC, self.__iv)
        ptx = cipher.decrypt(ctx)
        return ptx

    def _init_cm(self, save_spaces):
        """ Initialize the cipher machine for RealAes. Returns None as FakeAes does not require a cipher machine """
        return None

    def _gen_key(self, strategy, order, partial_size, prefix):
        """ Generate an empty key as RealAes does not require a key """
        return ""


class Plain(Cipher):
    def __init__(self, alphabet, save_spaces=True):
        super().__init__(alphabet, "plain", save_spaces)

    def _gen_key(self, strategy, order, partial_size, prefix):
        """ Generate a key to mask plaintexts to correspond to correct partial decryptions """
        # Mask plain texts according to char frequency for each message separately
        key = list(self.alphabet)[1:]
        key_size = len(key)

        idx_unmask = [key.index(order[i]) for i in range(partial_size)]
        key_masked = [self.alphabet[0] for _ in range(key_size)]
        for idx in idx_unmask:
            key_masked[idx] = key[idx]
        key_masked.insert(0, self.alphabet[0])
        return "".join(key_masked)

    def decrypt(self, msg, key):
        """ Decryption of plaintexts should not be possible """
        raise NotImplementedError()

    def _init_cm(self, save_spaces):
        """ Initialize a cipher machine for the plain cipher """
        cm = secretpy.CryptMachine(
            secretpy.SimpleSubstitution(), alphabet=self.alphabet
        )
        if save_spaces:
            cm = secretpy.SaveSpaces(cm)
        return cm


class Caesar(Cipher):
    def __init__(self, alphabet, save_spaces=True):
        super().__init__(alphabet, "caesar", save_spaces)

    def _gen_key(self, strategy, order, partial_size, prefix):
        """ Generate a key for the caesar cipher """
        # Special char should not be considered in caesar encryption
        key = randint(1, len(self.alphabet) - 1)
        return key

    def encrypt(self, msg, key):
        """ Encrypt a message under the caesar cipher """
        key = int(key)
        return super().encrypt(msg, key)

    def decrypt(self, msg, key):
        """ Decrypt a message encrypted under the caesar cipher """
        key = int(key)
        return super().decrypt(msg, key)

    def _init_cm(self, save_spaces):
        """ Initilize a cipher machine for the caesar cipher """
        # Don't include special char in caesar encryption
        cm = secretpy.CryptMachine(secretpy.Caesar(), alphabet=self.alphabet[1:])
        if save_spaces:
            cm = secretpy.SaveSpaces(cm)
        return cm


class Substitution(Cipher):
    def __init__(self, alphabet, save_spaces=True):
        super().__init__(alphabet, "substitution", save_spaces)

    def _gen_key(self, strategy, order, partial_size, prefix):
        """ Key generation for the substitution cipher is only done in batches.
        See gen_keys """
        raise NotImplementedError()

    def gen_data(
        self,
        msgs,
        order,
        key_rot=1,
        key_gen_strategy="random",
        partial_size=0,
        low_dist_data=False,
    ):
        """ Generate data for the class incorrect decryption """
        encs, keys = self.encrypt_all(
            msgs,
            order,
            key_rot=key_rot,
            key_gen_strategy=key_gen_strategy,
            partial_size=partial_size,
        )
        dec_store = []
        key_store = []
        for enc, key, msg in zip(encs, keys, msgs):
            if low_dist_data and partial_size > 6:
                keys_dec = self.gen_low_dist_key_shuffle(
                    key, "random", order, partial_size=partial_size, char_space=enc
                )
            else:
                keys_dec = self.gen_keys(
                    1, "random", order, partial_size=partial_size, char_space=enc
                )
            key_dec = keys_dec[0]
            dec = self.decrypt(enc, key_dec)
            if dec != msg:
                dec_store.append(dec)
                key_store.append(key_dec)
            else:
                dec_store.append(self.alphabet[0])
                key_store.append(self.alphabet[0])
        return dec_store, key_store

    def gen_low_dist_key_prefix(
        self,
        key_correct,
        strategy,
        order,
        partial_size=0,
        char_space=None,
        partial_step=8,
    ):
        """ Generate additional samples of the class incorrect decryption with low distance to the correct decryption.
        The generation of the decryption relies on fixing a correct prefix of the key up to a specific length and only randomize
        following characters. """
        
        fixed = max(partial_size - partial_step, 0)
        # Mask key except for part that works as prefix
        prefix = list(key_correct)
        for i in range(fixed, len(order)):
            idx = self.alphabet.index(order[i])
            prefix[idx] = self.alphabet[0]
        prefix = "".join(prefix)

        keys_dec = self.gen_keys(
            1,
            "random",
            order,
            partial_size=partial_size,
            prefix=prefix,
            char_space=char_space,
        )
        return keys_dec

    def gen_low_dist_key_shuffle(
        self,
        key_correct,
        strategy,
        order,
        partial_size=0,
        char_space=None,
        partial_step=8,
    ):
        """ Generate additional samples of the class incorrect decryption with low distance to the correct decryption.
        The generation of the decryption relies on shuffling a random number of characters of the key. """
        
        n_shuffles = random.randint(1, partial_step)
        list_key = list(key_correct)
        for i in range(n_shuffles):
            # Get index of random high frequency character
            random_upper = min(len(self.alphabet) - 1, partial_step + i)
            idx1 = random.randint(0, random_upper)
            idx1 = self.alphabet.index(order[idx1])
            
            # Get index of random key to swap with
            idx2 = random.randint(1, len(self.alphabet) - 1)
            if self.alphabet[idx2] in key_correct:
                list_key[idx1], list_key[idx2] = list_key[idx2], list_key[idx1]
            else:
                list_key[idx1] = self.alphabet[idx2]

            
        new_key = "".join(list_key)
        return [new_key]

    def gen_keys(
        self, n_keys, strategy, order, partial_size=0, prefix="", char_space=None
    ):
        
        """ Generate a batch of keys for the substitution cipher. """
        if partial_size == 0:
            partial_size = len(self.alphabet) - 1

        if char_space is None:
            char_space = self.alphabet[1:]
        else:
            # Remove duplicates and special char from char space
            char_space = "".join(set(char_space))
            char_space = char_space.replace(self.alphabet[0], "")
            char_space = char_space.replace(" ", "")

        key_size = len(self.alphabet[1:])

        prefix_stripped = prefix.replace(self.alphabet[0], "")
        order = order.replace(self.alphabet[0], "")

        # Remove previously chosen characters from key
        for c in prefix_stripped:
            char_space = char_space.replace(c, "")

        assert (
            len(prefix_stripped) <= partial_size
        ), "Prefix must be smaller than unmasked part of the desired key"
        offset = len(prefix_stripped)

        if prefix == "":
            key_base = [self.alphabet[0] for _ in range(key_size)]
        else:
            key_base = list(prefix)[1:]

        # Get indices of the relevant chars of key
        idx_shuffle = []
        i = 0
        for i in range(offset, partial_size):
            idx_shuffle.append(self.alphabet.index(order[i]) - 1)

        keys = []

        if strategy == "iterative":
            keys = self._list_keys(key_base, idx_shuffle, char_space)

        elif strategy == "random":
            for i in range(n_keys):
                char_space = list(char_space)
                shuffle(char_space)
                shuffle(idx_shuffle)
                key_masked = copy.deepcopy(key_base)
                for val_idx, key_idx in enumerate(idx_shuffle):
                    if len(char_space) > val_idx:
                        key_masked[key_idx] = char_space[val_idx]

                key_masked.insert(0, self.alphabet[0])
                keys.append("".join(key_masked))
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        return keys

    def opposite(self, key):
        """Compute the opposite key.

        Definition of opposite key:

        if opposite(key) -> key_ and enc(msg, key) -> enc
        then enc(enc, key_) -> msg
        """

        key_dict = {}
        for k, a in zip(key, self.alphabet):
            key_dict[k] = a
        opp = self.alphabet[0]
        for a in self.alphabet[1:]:
            if a in key:
                opp += key_dict[a]
            else:
                opp += self.alphabet[0]
        return opp

    def decrypt(self, msg, key):
        """ Decrypt a message encrypted under the substitution cipher. """
        # Decryption is equal to encryption with the opposite key.
        # This hack solves the problem of mapping the special char to many other chars in partial decryption.
        opp_key = self.opposite(key)
        return self.encrypt(msg, opp_key)

    def _init_cm(self, save_spaces):
        """ Initialize a cipher machine for the substitution cipher. """
        
        cm = secretpy.CryptMachine(
            secretpy.SimpleSubstitution(), alphabet=self.alphabet
        )
        if save_spaces:
            cm = secretpy.SaveSpaces(cm)
        return cm

    def _list_keys(self, base_key, base_idx, chars):
        """List all possible keys

        Args:
            base_key (list): Starting key

            base_idx (list): List of positions to vary the key

            chars (string): Character set that is used to vary the key
        """

        keys = []
        idx_shuffle = copy.deepcopy(base_idx)
        if len(idx_shuffle) == 1:
            idx = idx_shuffle[0]
            for c in chars:
                key_new = copy.deepcopy(base_key)
                key_new[idx] = c
                key_new.insert(0, self.alphabet[0])
                keys.append("".join(key_new))
        else:
            idx = idx_shuffle.pop(0)
            for c in chars:
                key_new = copy.deepcopy(base_key)
                new_chars = copy.deepcopy(chars)
                new_chars = new_chars.replace(c, "")
                key_new[idx] = c
                keys.extend(self._list_keys(key_new, idx_shuffle, new_chars))
        return keys


class Vigenere(Cipher):
    def __init__(self, alphabet, save_spaces=True):
        super().__init__(alphabet, "vigenere", save_spaces)

    def _gen_key(self, strategy, order, partial_size, prefix):
        """ Generate a key for the vigenere cipher """
        
        key = list(self.alphabet)
        shuffle(key)
        l = randint(1, len(self.alphabet))
        key = "".join(key[:l])
        return key

    def _init_cm(self, save_spaces):
        """ Initialize a cipher machine for the vigenere cipher """
        
        cm = secretpy.CryptMachine(secretpy.Vigenere(), alphabet=self.alphabet)
        if save_spaces:
            cm = secretpy.SaveSpaces(cm)
        return cm


class Columnar(Cipher):
    def __init__(self, alphabet, save_spaces=True):
        super().__init__(alphabet, "columnar", save_spaces)

    def _gen_key(self, strategy, order, partial_size, prefix, encryption=True):
        """ Generate a key for the columnar cipher """
        
        key = list(self.alphabet)
        shuffle(key)
        l = randint(1, len(self.alphabet))
        key = "".join(key[:l])
        return key

    def _init_cm(self, save_spaces):
        """ Initialize a cipher machine for the columnar cipher """
        
        cm = secretpy.CryptMachine(
            secretpy.ColumnarTransposition(), alphabet=self.alphabet
        )
        if save_spaces:
            cm = secretpy.SaveSpaces(cm)
        return cm


def gen_cipher_dict(alphabet, save_spaces=True):
    """ Generate a dictionary containing cipher machines for all possible ciphers"""

    cipher_dict = {
        "plain": Plain(alphabet, save_spaces=save_spaces),
        "caesar": Caesar(alphabet, save_spaces=save_spaces),
        "substitution": Substitution(alphabet, save_spaces=save_spaces),
        "vigenere": Vigenere(alphabet, save_spaces=save_spaces),
        "columnar": Columnar(alphabet, save_spaces=save_spaces),
        "realAes": RealAes(alphabet, save_spaces=save_spaces),
        "fakeAes": FakeAes(alphabet, save_spaces=save_spaces),
    }
    return cipher_dict
