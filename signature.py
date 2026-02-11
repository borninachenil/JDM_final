from dataclasses import dataclass, field

from config import TRT_RELATION_IDS

@dataclass
class Signature:
    hyperonyms: set = field(default_factory=set)
    trt: set = field(default_factory=set)
    sst: set = field(default_factory=set)

    def to_set(self):
        symbols = set()
        for h in self.hyperonyms:
            symbols.add(f"H:{h}")
        for t in self.trt:
            symbols.add(f"TRT:{t}")
        for s in self.sst:
            symbols.add(f"SST:{s}")
        return symbols

    def __len__(self):
        return len(self.hyperonyms) + len(self.trt) + len(self.sst)


class SignatureExtractor:
    def __init__(self, jdm_client, max_hyperonyms=20):
        self.client = jdm_client
        self.max_hyperonyms = max_hyperonyms # On pourrait tweak la val pour voir 
        self._cache = {}

    def extract(self, word):
        if word in self._cache:
            return self._cache[word]

        sig = Signature()

        hyp = self.client.get_hyperonyms(word)
        if hyp:
            top_hyp = sorted(hyp.items(), key=lambda x: -x[1])[:self.max_hyperonyms]
            sig.hyperonyms = {name for name, _ in top_hyp}

        sem = self.client.get_infosem(word)
        sig.sst = set(sem.keys())
        all_types = self.client.get_relation_types_present(word)
        sig.trt = {str(t) for t in all_types if t in TRT_RELATION_IDS}
        self._cache[word] = sig
        
        return sig

    def extract_pair(self, a, b):
        sig_a = self.extract(a)
        sig_b = self.extract(b)

        return sig_a.to_set(), sig_b.to_set()