# ck_filter_plasma_to_neutral.py
# usage: python ck_filter_plasma_to_neutral.py chem.inp chem_neutral.inp
import re, sys

BAN = {b.upper() for b in ["E", "e-", "PHOTON"]}

def has_banned_species(eq_line: str) -> bool:
    s = re.split(r"[;!]", eq_line, 1)[0]
    s = s.replace("<=>","=").replace("=>","=")
    if "=" not in s:
        return False
    lhs, rhs = s.split("=", 1)
    def tokens(part):
        t = []
        for ch in part.split("+"):
            ch = ch.strip()
            ch = re.sub(r"^[0-9.+-Ee]*\s*", "", ch)
            if ch:
                t.append(ch.split()[0])
        return t
    species = {x.upper() for x in tokens(lhs) + tokens(rhs)}
    return any(x in BAN for x in species)

def main(fin, fout):
    with open(fin, "r") as f: lines = f.readlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        tag = line.strip().upper()
        if tag.startswith("SPECIES"):
            out.append(line)
            buf = []
            i += 1
            while i < len(lines) and lines[i].strip().upper() != "END":
                buf.append(lines[i]); i += 1
            txt = "".join(buf)
            words = re.split(r"(\s+|!.*\n)", txt)
            rebuilt = []
            for w in words:
                if not w or w.isspace() or w.startswith("!"):
                    rebuilt.append(w)
                else:
                    if w.strip().upper() in BAN:
                        continue
                    rebuilt.append(w)
            out.append("".join(rebuilt))
            out.append("END\n")
        elif tag.startswith("REACTIONS"):
            out.append(line)
            i += 1
            keep_current = True
            while i < len(lines) and lines[i].strip().upper() != "END":
                L = lines[i]
                if ("=" in L) or ("<=>" in L) or ("=>" in L):
                    keep_current = not has_banned_species(L)
                if keep_current:
                    out.append(L)
                i += 1
            out.append("END\n")
        else:
            out.append(line)
        i += 1
    with open(fout, "w") as g: g.writelines(out)
    print(f"[ok] wrote {fout}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python ck_filter_plasma_to_neutral.py chem.inp chem_neutral.inp")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
