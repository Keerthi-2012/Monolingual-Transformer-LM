"""Phase 3: programmatic Hindi comparative / transitive-reasoning dataset (ground truth known by construction).

Each example: context sentences + a question, a short reasoning chain (ordering), and a final answer.
    प्रश्न: <context> <question>
    तर्क: <ordering / numbers>। इसलिए ...
    उत्तर: <answer>

Task kinds : chain3 / chain4 (transitive ordering from shuffled pairwise statements),
             num2 / num3 (compare stated numbers).
Question   : superlative (largest / smallest), pair ("X या Y"), yes/no ("क्या X Y से ... है?").
Gender     : adjectives agree with the sentence subject (लंबा / लंबी, ...).

Leakage control (reported in stats):
  * entity names are DISJOINT across train / val / test (names and item words); training uses ~1,450
    names (real + made-up) so the model must copy names rather than memorize them;
  * test_ood additionally uses statement and question templates never seen in training;
  * exact duplicate prompts are removed within and across splits.

Usage: python generate.py --out data
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

# ------------------------------------------------------------------ lexicon
MALE = "राम श्याम मोहन सोहन रवि अजय विजय संजय राहुल रोहित अमित सुमित नितिन विकास प्रकाश दीपक मनीष सुरेश रमेश महेश गणेश दिनेश राजेश कमल अनिल सुनील अशोक मुकेश हरीश योगेश आदित्य अर्जुन कृष्ण करण वरुण तरुण अभय नवीन पवन सचिन".split()
FEMALE = "सीता गीता रीता मीना सुनीता अनीता कविता सविता ममता पूजा नेहा स्नेहा रेखा लता शोभा आशा उषा निशा दिशा ज्योति प्रिया सपना सरिता राधा काजल अंजली सोनी किरण पायल दीपिका मंजू सुमन ऋतु नीता रचना भारती शालिनी अदिति तनु बबीता".split()
ITEMS_M = "बैग फोन जूता छाता तकिया लैपटॉप पंखा दीया गिलास चश्मा घड़ा कुकर रेडियो टीवी कंप्यूटर रुमाल चम्मच पैन बर्तन कंबल".split()
ITEMS_F = "किताब कलम घड़ी कुर्सी मेज़ कमीज़ टोपी साइकिल थाली कंघी चटाई बाल्टी रस्सी लालटेन पेटी मिठाई गेंद चाबी दवाई पतंग".split()


# Masculine nouns in -ा take the oblique (-े) form before a postposition (से, की, का, में, की तुलना...).
OBL = {"जूता": "जूते", "छाता": "छाते", "तकिया": "तकिये", "चश्मा": "चश्मे", "घड़ा": "घड़े", "पंखा": "पंखे", "दीया": "दीये"}
O = lambda w: OBL.get(w, w)


def split_pool(words, n_test=5, n_val=5):
    """Disjoint pools: first n_test -> test, next n_val -> val, rest -> train."""
    return {"test": words[:n_test], "val": words[n_test:n_test + n_val], "train": words[n_test + n_val:]}


CONS = list("कखगचजतदनपबमरलवसहशधभय")
MATRAS = ["", "ा", "ि", "ी", "ु", "ू", "े", "ो"]


def synth_names(n, gender, seed, forbid):
    """n unique pronounceable made-up Hindi names (2-3 syllables), disjoint from `forbid`.

    Male names end in a bare consonant, female names in -ा or -ी, so adjective agreement stays natural.
    A large pool of such names forces the model to COPY names from the prompt instead of memorizing a few.
    """
    rng, out = random.Random(seed), []
    while len(out) < n:
        k = rng.choice([2, 3])
        syl = [rng.choice(CONS) + rng.choice(MATRAS) for _ in range(k - 1)]
        syl.append(rng.choice(CONS) + (rng.choice(["ा", "ी"]) if gender == "f" else ""))
        w = "".join(syl)
        if w not in forbid:
            forbid.add(w)
            out.append(w)
    return out


_used = set(MALE + FEMALE + ITEMS_M + ITEMS_F)
N_SYNTH = {"train": 700, "val": 25, "test": 40}      # per gender; every name is unique across splits
PEOPLE = {}
for _s in ("train", "val", "test"):
    _m, _f = split_pool(MALE)[_s], split_pool(FEMALE)[_s]
    _sm = synth_names(N_SYNTH[_s], "m", {"train": 11, "val": 12, "test": 13}[_s], _used)
    _sf = synth_names(N_SYNTH[_s], "f", {"train": 21, "val": 22, "test": 23}[_s], _used)
    PEOPLE[_s] = [(n, "m") for n in _m + _sm] + [(n, "f") for n in _f + _sf]
THINGS = {s: [(n, "m") for n in split_pool(ITEMS_M, 4, 4)[s]] + [(n, "f") for n in split_pool(ITEMS_F, 3, 3)[s]] for s in ("train", "val", "test")}

# attribute: more/less adjective (masc, fem), noun, noun gender (for की/का), unit, numeric range, entity kind
ATTRS = {
    "height": dict(more=("लंबा", "लंबी"), less=("छोटा", "छोटी"), noun="ऊँचाई", ng="f", unit="सेंटीमीटर", rng=(120, 195), ent="p"),
    "age": dict(more=("बड़ा", "बड़ी"), less=("छोटा", "छोटी"), noun="उम्र", ng="f", unit="साल", rng=(5, 80), ent="p"),
    "weight": dict(more=("भारी", "भारी"), less=("हल्का", "हल्की"), noun="वज़न", ng="m", unit="किलो", rng=(30, 110), ent="p"),
    "speed": dict(more=("तेज़", "तेज़"), less=("धीमा", "धीमी"), noun="रफ़्तार", ng="f", unit="किलोमीटर प्रति घंटा", rng=(5, 40), ent="p"),
    "price": dict(more=("महँगा", "महँगी"), less=("सस्ता", "सस्ती"), noun="कीमत", ng="f", unit="रुपये", rng=(20, 5000), ent="i"),
}

# ------------------------------------------------------------------ templates
# statement of fact "hi has more than lo": S1/S3 train, S2/S4 only in test_ood
STMT = {
    "S1": lambda hi, lo, adj: f"{hi} {O(lo)} से {adj(hi, 'more')} है।",
    "S3": lambda hi, lo, adj: f"{lo} {O(hi)} से {adj(lo, 'less')} है।",
    "S2": lambda hi, lo, adj: f"{O(hi)} की तुलना में {lo} {adj(lo, 'less')} है।",
    "S4": lambda hi, lo, adj: f"{O(lo)} की अपेक्षा {hi} {adj(hi, 'more')} है।",
}
STMT_SETS = {"iid": ["S1", "S3"], "ood": ["S2", "S4"]}
# superlative questions ({a}=adjective); people use "कौन", things use "चीज़ कौन सी"
SUP_Q = {
    "p": {"iid": ["सबसे {a} कौन है?", "इनमें सबसे {a} कौन है?", "इन सब में से कौन सबसे {a} है?"],
          "ood": ["बताइए, सबसे {a} कौन है?", "इन सभी में सबसे {a} कौन है?"]},
    "i": {"iid": ["सबसे {a} चीज़ कौन सी है?", "इनमें सबसे {a} चीज़ कौन सी है?"],
          "ood": ["बताइए, सबसे {a} चीज़ कौन सी है?", "इन सभी में सबसे {a} चीज़ कौन सी है?"]},
}
PAIR_Q = {"iid": ["{x} और {y} में से कौन {a} है?"], "ood": ["{x} या {y}, कौन {a} है?"]}
YN_Q = {"iid": ["क्या {x} {y} से {a} है?"], "ood": ["क्या यह सच है कि {x} {y} से {a} है?"]}
FACT = lambda e, at, n: f"{O(e)} {'का' if ATTRS[at]['ng'] == 'm' else 'की'} {ATTRS[at]['noun']} {n} {ATTRS[at]['unit']} है।"


# ------------------------------------------------------------------ generation
class Maker:
    def __init__(self, split, style, rng):
        self.split, self.style, self.rng = split, style, rng      # style: "iid" or "ood" templates

    def adj_fn(self, gender, attr):
        """adj(entity, 'more'|'less') with gender agreement."""
        def adj(ent, d):
            return ATTRS[attr][d][0 if gender[ent] == "m" else 1]
        return adj

    def entities(self, attr, k):
        pool = PEOPLE[self.split] if ATTRS[attr]["ent"] == "p" else THINGS[self.split]
        return self.rng.sample(pool, k)

    def make(self, kind):
        rng = self.rng
        attr = rng.choice(list(ATTRS))
        k = {"chain3": 3, "chain4": 4, "num2": 2, "num3": 3}[kind]
        ents = self.entities(attr, k)
        gender = dict(ents)
        names = [n for n, _ in ents]
        adj = self.adj_fn(gender, attr)
        A = ATTRS[attr]
        qtype = rng.choice(["sup", "sup", "yn", "pair"] if kind == "num2" else ["sup", "sup", "yn"])
        d = rng.choice(["more", "less"])
        if kind.startswith("chain"):
            order = names[:]                                     # order[0] has the MOST of the attribute
            tmpl = [rng.choice(STMT_SETS[self.style]) for _ in range(k - 1)]
            sents = [STMT[t](order[i], order[i + 1], adj) for i, t in enumerate(tmpl)]
            rng.shuffle(sents)
            vals = None
            used = tmpl
        else:
            lo, hi = A["rng"]
            tie = k == 2 and qtype != "sup" and rng.random() < 0.1   # ties make "the largest" ambiguous
            nums = rng.sample(range(lo, hi + 1), k)
            if tie:
                nums[1] = nums[0]
            sents = [FACT(n, attr, v) for n, v in zip(names, nums)]
            vals = dict(zip(names, nums))
            order = sorted(names, key=lambda n: -vals[n])         # stable; ties keep input order
            used = ["FACT"]
        rank = {n: i for i, n in enumerate(order)}              # smaller = more of the attribute
        if qtype == "sup":
            a_m = A[d][0] if A["ent"] == "p" else A[d][1]       # things take the feminine form with "चीज़"
            q = rng.choice(SUP_Q[A["ent"]][self.style]).format(a=a_m)
            ans = order[0] if d == "more" else order[-1]
            tail = (f"इसलिए सबसे {A[d][0]} {ans} है।" if A["ent"] == "p"
                    else f"इसलिए सबसे {A[d][1]} चीज़ {ans} है।")
        elif qtype == "pair":
            x, y = names
            q = rng.choice(PAIR_Q[self.style]).format(x=x, y=y, a=A[d][0])
            if self.style == "iid" and A["ent"] == "i":   # "X और Y में से": both take the oblique form
                q = q.replace(f"{x} और {y}", f"{O(x)} और {O(y)}")
            if vals[x] == vals[y]:
                ans = "दोनों बराबर हैं"
                tail = "इसलिए दोनों बराबर हैं।"
            else:
                ans = max(names, key=lambda n: vals[n]) if d == "more" else min(names, key=lambda n: vals[n])
                tail = f"इसलिए {A[d][0]} {ans} है।"
        else:
            x, y = rng.sample(names, 2)
            q = rng.choice(YN_Q[self.style]).format(x=x, y=O(y), a=adj(x, d))   # y is the object of "से"
            truth = rank[x] < rank[y] if d == "more" else rank[x] > rank[y]
            if vals is not None and vals[x] == vals[y]:
                truth = False
            ans = "हाँ" if truth else "नहीं"
            tail = f"इसलिए उत्तर {ans} है।"
        if vals is not None:
            facts = ", ".join(f"{n} {vals[n]}" for n in names)
            cot = f"{A['noun']}: {facts}। क्रम (ज़्यादा से कम): {' > '.join(order)}। {tail}"
        else:
            cot = f"{A['noun']} का क्रम (ज़्यादा से कम): {' > '.join(order)}। {tail}"
        context = " ".join(sents)
        return {"kind": kind, "attr": attr, "qtype": qtype, "templates": used, "prompt": f"प्रश्न: {context} {q}\nतर्क:",
                "cot": " " + cot, "answer": ans}


def build_split(split, style, n, seed, seen):
    """Generate n unique examples; `seen` (shared across splits) blocks duplicate prompts."""
    rng = random.Random(seed)
    mk = Maker(split, style, rng)
    kinds = ["chain3", "chain4", "num2", "num3"]
    out, tries = [], 0
    while len(out) < n and tries < n * 50:
        tries += 1
        ex = mk.make(kinds[len(out) % len(kinds)])
        if ex["prompt"] in seen:
            continue
        seen.add(ex["prompt"])
        ex["id"], ex["split"] = f"{split}_{style}_{len(out)}", f"{split}_{style}"
        out.append(ex)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1500)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    seen = set()
    sets = {"train": build_split("train", "iid", a.n_train, 1, seen),
            "val": build_split("val", "iid", a.n_val, 2, seen),
            "test_iid": build_split("test", "iid", a.n_test, 3, seen),      # new names, seen templates
            "test_ood": build_split("test", "ood", a.n_test, 4, seen)}     # new names AND new templates
    stats = {}
    for name, rows in sets.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = {"n": len(rows), "kind": Counter(r["kind"] for r in rows), "qtype": Counter(r["qtype"] for r in rows),
                       "attr": Counter(r["attr"] for r in rows), "answer_yes": sum(r["answer"] == "हाँ" for r in rows),
                       "answer_no": sum(r["answer"] == "नहीं" for r in rows)}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v["n"] for k, v in stats.items()}))
