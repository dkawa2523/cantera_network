# make_reactions_from_csv.py
# Usage: python make_reactions_from_csv.py reactions.csv > reactions_block.txt
import csv, sys

def fmt_row(eq,A,b,Ta,units):
    note = ""
    return f"{eq:<35s}  {A: .3E}  {b: .3f}  {Ta: .1f}{note}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_reactions_from_csv.py reactions.csv", file=sys.stderr); sys.exit(1)
    with open(sys.argv[1], newline="") as f:
        rdr = csv.DictReader(f)
        print("REACTIONS    KCAL/MOLE  MOLES  CM  SEC")
        for r in rdr:
            print(fmt_row(r["equation"], float(r["A"]), float(r["b"]),
                          float(r["Ea_over_R"]), r.get("units","")))
        print("END")
