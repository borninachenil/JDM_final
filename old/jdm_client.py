"""Client API JeuxDeMots avec cache disque."""

import json
import os
import time
import hashlib
import urllib.error
import urllib.parse
import urllib.request

from config import JDM_API_BASE, API_RATE_LIMIT, API_REQUEST_LIMIT, CACHE_DIR


class JDMClient:
    """Client pour l'API JeuxDeMots avec cache disque et memoire."""

    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        self._memory_cache = {}
        self._last_request_time = 0
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, word, types_ids, min_weight):
        """Genere une cle de cache unique."""
        raw = f"{word}|{types_ids}|{min_weight}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.json")

    def _load_from_cache(self, key):
        if key in self._memory_cache:
            return self._memory_cache[key]
        path = self._cache_path(key)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Convertir les cles de nodes en int (JSON les serialise en string)
            if "nodes" in data and isinstance(data["nodes"], dict):
                data["nodes"] = {int(k): v for k, v in data["nodes"].items()}
            self._memory_cache[key] = data
            return data
        return None

    def _save_to_cache(self, key, data):
        self._memory_cache[key] = data
        path = self._cache_path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < API_RATE_LIMIT:
            time.sleep(API_RATE_LIMIT - elapsed)
        self._last_request_time = time.time()

    def _api_call(self, endpoint, params=None, retries=3):
        """Appel HTTP GET vers l'API JDM avec retry."""
        url = f"{JDM_API_BASE}{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        for attempt in range(retries):
            self._rate_limit()

            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")

            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if attempt < retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return None
            except Exception:
                if attempt < retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return None

    def get_relations(self, word, types_ids=None, min_weight=0, limit=API_REQUEST_LIMIT):
        """
        Recupere les relations sortantes d'un mot.

        Retourne un dict structure :
        {
            "nodes": {node_id: {"name": str, "type": int, "weight": float}},
            "relations": [{"node1": int, "node2": int, "type": int, "weight": float}]
        }
        """
        types_str = str(types_ids) if types_ids is not None else "all"
        cache_key = self._cache_key(word, types_str, min_weight)

        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        encoded_word = urllib.parse.quote(word, safe="")
        params = {"min_weight": min_weight, "limit": limit}
        if types_ids is not None:
            params["types_ids"] = types_ids

        raw = self._api_call(f"/relations/from/{encoded_word}", params)

        if raw is None:
            result = {"nodes": {}, "relations": []}
            self._save_to_cache(cache_key, result)
            return result

        # Parser les noeuds
        nodes = {}
        for node in raw.get("nodes", []):
            nodes[node["id"]] = {
                "name": node.get("name", ""),
                "type": node.get("type", 0),
                "weight": node.get("w", 0),
            }

        # Parser les relations
        relations = []
        for rel in raw.get("relations", []):
            relations.append({
                "node1": rel.get("node1", 0),
                "node2": rel.get("node2", 0),
                "type": rel.get("type", 0),
                "weight": rel.get("w", 0),
            })

        result = {"nodes": nodes, "relations": relations}
        self._save_to_cache(cache_key, result)
        return result

    def get_hyperonyms(self, word):
        """Recupere les hyperonymes (r_isa, type 6) d'un mot. Retourne {nom: poids}."""
        data = self.get_relations(word, types_ids=6, min_weight=1)
        nodes = data["nodes"]
        result = {}
        for rel in data["relations"]:
            if rel["type"] == 6 and rel["weight"] > 0:
                node_id = rel["node2"]
                if node_id in nodes:
                    result[nodes[node_id]["name"]] = rel["weight"]
        return result

    def get_infosem(self, word):
        """Recupere les types semantiques (_INFO-SEM) d'un mot. Retourne {nom: poids}."""
        data = self.get_relations(word, types_ids=36, min_weight=1)
        nodes = data["nodes"]
        result = {}
        for rel in data["relations"]:
            if rel["type"] == 36 and rel["weight"] > 0:
                node_id = rel["node2"]
                if node_id in nodes:
                    name = nodes[node_id]["name"]
                    if name.startswith("_INFO-SEM"):
                        result[name] = rel["weight"]
        return result

    def get_relation_types_present(self, word):
        """Recupere l'ensemble des types de relations dans lesquels le mot apparait."""
        data = self.get_relations(word, types_ids=None, min_weight=0)
        types_present = set()
        for rel in data["relations"]:
            if rel["weight"] > 0:
                types_present.add(rel["type"])
        return types_present

    def prefetch_batch(self, words, progress=True):
        """Pre-charge les donnees JDM pour un ensemble de mots."""
        words = list(set(words))
        total = len(words)
        fetched = 0
        cached = 0

        for i, word in enumerate(words):
            # Verifier si deja en cache pour les 3 types de requetes
            key_isa = self._cache_key(word, "6", 1)
            key_sem = self._cache_key(word, "36", 1)
            key_all = self._cache_key(word, "all", 0)

            all_cached = all(
                self._load_from_cache(k) is not None
                for k in [key_isa, key_sem, key_all]
            )

            if all_cached:
                cached += 1
            else:
                self.get_relations(word, types_ids=6, min_weight=1)
                self.get_relations(word, types_ids=36, min_weight=1)
                self.get_relations(word, types_ids=None, min_weight=0)
                fetched += 1

            if progress and (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] {fetched} fetched, {cached} cached")

        if progress:
            print(f"  Prefetch termine : {fetched} fetched, {cached} cached sur {total} mots")


if __name__ == "__main__":
    client = JDMClient()

    print("=== Test: hyperonymes de 'desert' ===")
    hyp = client.get_hyperonyms("désert")
    for name, weight in sorted(hyp.items(), key=lambda x: -x[1]):
        print(f"  {name}: {weight}")

    print("\n=== Test: _INFO-SEM de 'désert' ===")
    sem = client.get_infosem("désert")
    for name, weight in sorted(sem.items(), key=lambda x: -x[1]):
        print(f"  {name}: {weight}")

    print("\n=== Test: types de relations presents pour 'désert' ===")
    types = client.get_relation_types_present("désert")
    print(f"  Types: {sorted(types)}")
