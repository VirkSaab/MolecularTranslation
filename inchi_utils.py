""" InChI string utilities.
Author: Jitender Singh Virk (virksaab.github.io)
Last Updated: 22 Apr, 2021
"""
import re, torch
import pandas as pd
from collections import defaultdict
from elements import elements
from fastprogress import progress_bar
from typing import Union

__all__ = [
    "LAYERS_SEQ", "dissect_inchi", "get_layerwise_re_patterns", "inchi_tokenizer",
    "VocabONE", "Tokenizer"
]

LAYERS_SEQ = ('main_layer', 'c_layer', 'h_layer', 'b_layer', 't_layer', 'm_layer', 's_layer', 'i_layer')


def dissect_inchi(inchi_string:str, patterns_dict:dict) -> Union[dict, str]:
    # DISSECT INCHI
    inchi_string = inchi_string.replace("InChI=1S/", '')
    inchi_string = inchi_string.split("/")
    inchi_dissected_dict = {}
    inchi_dissected_dict["main_layer"] = inchi_string.pop(0)
    for layer_name in LAYERS_SEQ[1:]: # No need to check main_layer
        for idx, sub_string in enumerate(inchi_string):
            if layer_name[0] == sub_string[0]:
                inchi_dissected_dict[layer_name] = inchi_string.pop(idx)[1:]
                break

    # MAKE TOKENS
    for layer_name, inchi_substring in inchi_dissected_dict.items():        
        tokens = re.split(patterns_dict[layer_name], inchi_dissected_dict[layer_name])
        tokens = list(filter(None, tokens))
        inchi_dissected_dict[layer_name] = tokens
        
    return inchi_dissected_dict


def get_layerwise_re_patterns() -> dict:
    """Patterns to use with re.split for each inchi string's sublayer"""
    patterns_dict = {}
    
    # MAIN LAYER
    data = elements.Elements
    # All elements in periodic table
    elems = sorted(data, key=lambda i:i.AtomicNumber)  # Based on their AtomicNumber
    # Sort longer names first for regex pattern formation
    element_symbols = sorted([e.Symbol for e in elems], key=lambda e: len(e), reverse=True)
    # Create regex pattern
    patterns_dict["main_layer"] = f"({'|'.join([f'{e}' for e in element_symbols])})" #f"({'|'.join([f'{e}[0-9]*' for e in element_symbols])})"
    
    # c LAYER
    patterns_dict["c_layer"] = r"([\d]*)"
    # h LAYER
    patterns_dict["h_layer"] = r"([-?,)(]|H[0-9]*)"
    # b LAYER
    patterns_dict["b_layer"] = r"([-+,])"
    # t LAYER
    patterns_dict["t_layer"] = r"([-+,])"
    # m LAYER
    patterns_dict["m_layer"] = r""
    # s LAYER
    patterns_dict["s_layer"] = r""
    # i LAYER
    patterns_dict["i_layer"] = r"([\d,]|D[0-9]*)"
    return patterns_dict

def inchi_tokenizer(inchi_string:str, patterns_dict:dict) -> list:
    dissected_dict = dissect_inchi(inchi_string, patterns_dict)
    tokens = []
    for layer_name, subtokens in dissected_dict.items():
        if layer_name == "main_layer":
            tokens += subtokens
        elif layer_name in LAYERS_SEQ[1:]:
            tokens += [f"/{layer_name[0]}"] + subtokens
        else:
            raise NameError(f"<layer_name> {layer_name} is not understood :(")
    return tokens


class VocabONE:
    def __init__(self, vocab:list, max_len:int, patterns_dict:dict, add_special_tokens:bool=True) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.patterns_dict = patterns_dict
        
        if add_special_tokens:
            self.pad_token, self.unk_token, self.bos_token, self.eos_token = "<pad>", "<unk>", "<bos>", "<eos>"
            self.ctoi_dict = {}
            if self.pad_token not in self.vocab:
                self.vocab = [self.pad_token, self.unk_token,self.bos_token,self.eos_token] + self.vocab
            self.max_len += 2
            self.ctoi_dict = defaultdict(self.handle_unk_char, {c:i for i, c in enumerate(self.vocab)})
                
    def handle_unk_char(self):
        return self.vocab.index(self.unk_token)
        
    def ctoi(self, c:str) -> int:
        return self.ctoi_dict[c]
    
    def itoc(self, i:int) -> str:
        return self.vocab[i]
    
    def __len__(self):
        return len(self.vocab)
    
    def save_vocab(self, path:str) -> None:
        torch.save({"vocab": self.vocab, "max_len": self.max_len, "patterns_dict": self.patterns_dict}, path)
        print("Saved @", path)
        
    @classmethod
    def from_inchi_pandas_column(cls, inchi_data:pd.Series, add_special_tokens:bool=True) -> list:
        patterns_dict = get_layerwise_re_patterns()
        max_len, vocab = 0, []
        for inchi_string in progress_bar(inchi_data.tolist()):
            tokens = inchi_tokenizer(inchi_string, patterns_dict)
            tokens_len = len(tokens)
            if tokens_len > max_len: 
                max_len = tokens_len
            vocab += list(set(tokens))
        vocab = sorted(list(set(vocab)))
        return cls(vocab, max_len, patterns_dict, add_special_tokens)
    
    @classmethod
    def from_file(cls, path:str, add_special_tokens:bool=True) -> object:
        vocab = torch.load(path)
        vocab, max_len, patterns_dict = vocab["vocab"], vocab["max_len"], vocab["patterns_dict"]
        return cls(vocab, max_len, patterns_dict, add_special_tokens)
    
    
class Tokenizer:
    def __init__(self, vocab:VocabONE=None) -> None:
        self.vocab = vocab # Vocab class instance
        
    def tokenize(self, inchi_string:str) -> list:
        return inchi_tokenizer(inchi_string, self.vocab.patterns_dict)
    
    def encode(self, inchi_string:str, pad_to_max_len:bool=True, pad_to_custom_len:int=None, verbose:bool=False) -> dict[list]:
        tokens = self.tokenize(inchi_string)
        special_flag = False
        if hasattr(self.vocab, "bos_token"): # Add start and end special tokens
            tokens = [self.vocab.bos_token] + tokens + [self.vocab.eos_token]
            special_flag = True
        
        # Convert to numeric
        inp_seq = [self.vocab.ctoi(t) for t in tokens]
        attn_mask = [1] * len(inp_seq)
        cap_len = len(inp_seq)
        
        # Add padding for max lengths
        if pad_to_custom_len != None:
            if verbose and pad_to_max_len:
                print(f"Overriding max len padding to custom len ={pad_to_custom_len} padding. Sequences > {pad_to_custom_len} will be truncated.")
            extra_len = pad_to_custom_len - len(inp_seq)
            if extra_len < 0: # Truncate
                inp_seq = inp_seq[:pad_to_custom_len-1] + [self.vocab.ctoi(self.vocab.eos_token)] if special_flag else inp_seq[:pad_to_custom_len]
                attn_mask = attn_mask[:pad_to_custom_len]
            else:
                inp_seq += [self.vocab.ctoi(self.vocab.pad_token)] * extra_len
                attn_mask += [0] * extra_len
        elif pad_to_max_len:
            extra_len = self.vocab.max_len - len(inp_seq)
            inp_seq += [self.vocab.ctoi(self.vocab.pad_token)] * extra_len
            attn_mask += [0] * extra_len
        return {"inp_seq": inp_seq, "attn_mask": attn_mask, "cap_len": cap_len}
    
    def decode(self, inp_seq:Union[torch.Tensor, list], add_inchi_std:bool=True) -> str:
        if isinstance(inp_seq, torch.Tensor):
            inp_seq = inp_seq.cpu().detach().tolist()
        
        tokens = [self.vocab.itoc(t) for t in inp_seq]
        if self.vocab.bos_token in tokens:
            _from = tokens.index(self.vocab.bos_token) + 1 # 1 is for excluding start token from string
        else:
            _from = 0
        if self.vocab.eos_token in tokens:
            _to = tokens.index(self.vocab.eos_token)
        elif self.vocab.pad_token in tokens:
            _to = tokens.index(self.vocab.pad_token)
        else:
            _to = len(tokens)
        inchi_string = ''.join(tokens[_from:_to])
        if add_inchi_std:
            inchi_string = "InChI=1S/" + inchi_string
        return inchi_string
    
    @classmethod
    def from_file(cls, path:str, add_special_tokens:bool=True) -> object:
        vocab = torch.load(path)
        vocab, max_len, patterns_dict = vocab["vocab"], vocab["max_len"], vocab["patterns_dict"]
        vocab_obj = VocabONE(vocab, max_len, patterns_dict, add_special_tokens)
        return cls(vocab_obj)