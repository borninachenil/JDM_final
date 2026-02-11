"""Extraction des signatures semantiques depuis JeuxDeMots."""

from dataclasses import dataclass, field

from config import R_ISA, R_INFOPOT, TRT_RELATION_IDS
from jdm_client import JDMClient
from data_loader import detect_definiteness


@dataclass
class Signature:
    """Signature semantique d'un terme, composee de 3 blocs + DEF optionnel."""
    hyperonyms: set = field(default_factory=set)    # Bloc H : noms des hyperonymes
    trt: set = field(default_factory=set)            # Bloc TRT : IDs des types de relations presentes
    sst: set = field(default_factory=set)            # Bloc SST : noms des _INFO-SEM

    def to_set(self):
        """Convertit la signature en un ensemble unique de symboles."""
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
    """Extrait les signatures semantiques des termes depuis JDM."""

    def __init__(self, jdm_client, max_hyperonyms=20):
        self.client = jdm_client
        self.max_hyperonyms = max_hyperonyms
        self._cache = {}

    def extract(self, word):
        """Extrait la signature complete d'un mot."""
        if word in self._cache:
            return self._cache[word]

        sig = Signature()

        # Bloc H : Hyperonymes via r_isa (type 6) - top N par poids
        hyp = self.client.get_hyperonyms(word)
        if hyp:
            top_hyp = sorted(hyp.items(), key=lambda x: -x[1])[:self.max_hyperonyms]
            sig.hyperonyms = {name for name, _ in top_hyp}

        # Bloc SST : Types semantiques standards via r_infopot (type 36)
        sem = self.client.get_infosem(word)
        sig.sst = set(sem.keys())

        # Bloc TRT : Types de relations dans lesquels le mot apparait
        all_types = self.client.get_relation_types_present(word)
        sig.trt = {str(t) for t in all_types if t in TRT_RELATION_IDS}

        self._cache[word] = sig
        return sig

    def extract_pair(self, a, b, include_def=True):
        """
        Extrait les signatures pour une paire (A, B).

        Retourne (sig_A_set, sig_B_set) sous forme d'ensembles de symboles.
        """
        sig_a = self.extract(a)
        sig_b = self.extract(b)

        set_a = sig_a.to_set()
        set_b = sig_b.to_set()

        if include_def:
            def_type = detect_definiteness(b)
            set_b.add(f"DEF:{def_type}")

        return set_a, set_b


if __name__ == "__main__":
    client = JDMClient()
    extractor = SignatureExtractor(client)

    test_words = ["désert", "Algérie", "tabouret", "bois", "saucisse", "Toulouse"]

    for word in test_words:
        sig = extractor.extract(word)
        print(f"\n=== Signature de '{word}' ===")
        print(f"  H ({len(sig.hyperonyms)}): {sorted(sig.hyperonyms)[:10]}")
        print(f"  TRT ({len(sig.trt)}): {sorted(sig.trt)}")
        print(f"  SST ({len(sig.sst)}): {sorted(sig.sst)}")
        print(f"  Total symboles: {len(sig.to_set())}")

    print("\n=== Test paire 'désert d'Algérie' ===")
    set_a, set_b = extractor.extract_pair("désert", "Algérie")
    print(f"  sig_A ({len(set_a)} symboles)")
    print(f"  sig_B ({len(set_b)} symboles)")
