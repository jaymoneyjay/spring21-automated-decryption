import pandas as pd
import torch
import src.cipher as cipher
from tqdm import tqdm
import copy


class Cracker:
    """ Base class to provide functionality crack specific ciphers. """
    
    def __init__(self, cipher_id, alphabet, model):
        self.alphabet = alphabet
        self.model_dict = {}

        # Add model with dummy key for all ciphers which have no partial decryption
        # and thus only one model.
        if model is not None:
            self.model_dict[0] = model
        self.cipher = None
        self.id = cipher_id

    def likelihood(self, decs, partial_size=0):
        """Compute likelihood of decryptions"""
        
        preds = self.model_dict[partial_size].predict_all(
            decs, self.alphabet, batch_size=256
        )
        preds = torch.tensor(preds)
        preds_plain = preds[:, 0]
        return preds_plain

    @staticmethod
    def accuracy(k_pred, k_target):
        """ Compute accuracy for predictions. """
        
        n_errors = 0
        for k_, k in zip(k_pred, k_target):
            if k_ != k:
                n_errors += 1
        return (1 - n_errors / len(k_target)) * 100


class Caesar(Cracker):
    """ Class to provide functionality to crack the caesar cipher. """
    def __init__(self, alphabet, model):
        super().__init__("caesar", alphabet, model)
        self.cipher = cipher.Caesar(alphabet)

    def key_search(self, enc, n_top=1):
        """ Compute most probable key(s) for specified encryption. """
        
        keys = [i for i in range(len(self.alphabet))]
        encs = [enc for _ in keys]
        dec_store = self.cipher.decrypt_all(encs, keys)

        # Caesar cracker has no partial encryption and thus only one model
        p = self.model_dict[0].predict(dec_store, self.alphabet)
        df_pred = pd.DataFrame(p, columns=["p_plain", "p_cipher"])
        return df_pred.sort_values(by="p_plain", ascending=False).head(n_top).index

    def key_search_all(self, encs, k=1, probs=False):
        """ Compute most probable key(s) for all specified encryptions. """
        
        n_keys = len(self.alphabet)
        key_store = [i for _ in enumerate(encs) for i in range(n_keys)]
        enc_store = [m for m in encs for _ in range(n_keys)]
        dec_store = self.cipher.decrypt_all(enc_store, key_store)
        preds = self.model_dict[0].predict_all(dec_store, self.alphabet, batch_size=256)

        preds = torch.tensor(preds)
        preds_plain = preds[:, 0]
        preds_plain = preds_plain.view(-1, len(self.alphabet))
        top_preds = preds_plain.sort(dim=1, descending=True).indices[:, :k]
        top_probs = preds_plain.sort(dim=1, descending=True).values[:, :k]

        if probs:
            return top_preds, top_probs
        return top_preds


class Substitution(Cracker):
    """ Class to provide functionality to crack the substitution cipher. """
    
    def __init__(self, alphabet, model_dict):
        super().__init__("substitution", alphabet, None)
        self.cipher = cipher.Substitution(alphabet)
        self.model_dict = model_dict

    def top_keys(self, enc, keys, n, model_key, use_loss):
        """Compute most probable keys for the specified encryption out of a
        given set of keys """
        
        encs = [enc for _ in keys]
        decs = self.cipher.decrypt_all(encs, keys)

        if model_key not in self.model_dict.keys():
            raise ValueError(f"Cracker has no model with model key {model_key}")

        model = self.model_dict[model_key]
        preds = model.predict_all(decs, self.alphabet, use_loss, batch_size=256)
        preds = torch.tensor(preds)
        preds_plain = preds[:, 0]
        
        top_preds = preds_plain.sort(dim=0, descending=True).indices
        top_probs = preds_plain.sort(dim=0, descending=True).values
        
        if n != 0:
            top_preds = top_preds[:n]
            top_probs = top_probs[:n]
            
        top_keys = [keys[k.item()] for k in top_preds]
        top_probs = [p.item() for p in top_probs]
        df_top_keys = pd.DataFrame({"Key": top_keys, "P": top_probs})
        return df_top_keys

    def key_search(
        self,
        enc,
        order,
        use_loss=False,
        prefixes=None,
        strategy="random",
        n_top=10,
        n_candidates=10000,
        return_all=False
    ):
        """ Compute the most probable key(s) for a specified encryption. """
        
        if prefixes is None:
            prefixes = [""]

        df_keys = pd.DataFrame({"Key": prefixes, "P": [0 for _ in prefixes]})
        all_keys = []
        assert len(self.model_dict) > 0
        for l in tqdm(self.model_dict.keys()):
            df_keys_new = pd.DataFrame({"Key": [], "P": []})
            for key_ in df_keys.Key.values:
                candidates = self.cipher.gen_keys(
                    n_candidates,
                    strategy,
                    order,
                    partial_size=l,
                    prefix=key_,
                    char_space=None,
                )
                if len(candidates) < 1:
                    continue
                df_key_ = self.top_keys(enc, candidates, n_top, l, use_loss)
                df_keys_new = df_keys_new.append(df_key_)

            df_keys = copy.deepcopy(df_keys_new)
            df_keys = df_keys.drop_duplicates()
            df_keys = df_keys.sort_values(by=["P"], ascending=False)
            if n_top != 0:
                df_keys = df_keys.head(n_top)
            all_keys.append(df_keys)

        if return_all:
            return all_keys
        return df_keys
    
