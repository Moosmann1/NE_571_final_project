import pandas as pd

class CrossSectionVocabulary:
    def __init__(self, csv_file):
        # Load CSV directly
        self.df = pd.read_csv(csv_file)

        # Build vocabulary as dictionary on the fly
        self.vocab = {}
        for _, row in self.df.iterrows():
            key = (row["BURNUP"], row["TF"], row["TM"], row["BOR"], row["GROUP"])
            xs_data = {
                "ABSORPTION": row["ABSORPTION"],
                "CAPTURE": row["CAPTURE"],
                "FISSION": row["FISSION"],
                "NU-FISSION": row["NU-FISSION"],
                "TRANSPORT": row["TRANSPORT"],
                "OUT-SCATTER": row["OUT-SCATTER"],
                "DIFF(1/3TR)": row["DIFF(1/3TR)"],
            }
            self.vocab[key] = xs_data

    def get(self, tf, tm, bor, burnup, group):
        key = (burnup, tf, tm, bor, group)
        if key in self.vocab:
            return self.vocab[key]
        else:
            raise KeyError(f"No cross section data found for: BURNUP={burnup}, TF={tf}, TM={tm}, BOR={bor}, GROUP={group}")

    def list_keys(self):
        return list(self.vocab.keys())

# ------------------------------
# Example usage
"""
# Create instance from CSV
vocab = CrossSectionVocabulary("XS.csv")

# Define your query parameters
burnup = 57.5
tf = 1242.0
tm = 557.0
bor = 200.0
group = 2.0
#################     real Cross Section Data:
#################     ABSORPTION: 0.0837694
#################     CAPTURE: 0.0505907
#################     FISSION: 0.0331787
#################     NU-FISSION: 0.0950077
#################     TRANSPORT: 0.952356
#################     OUT-SCATTER: 0.00168031
#################     DIFF(1/3TR): 0.350009

# Define your query parameters
burnup = 57.5
tf = 1242.0
tm = 557.0
bor = 200.0
group = 2.0

# Call the get function (no try/except)
xs = vocab.get(burnup, tf, tm, bor, group)

# Print cross section data
print("\nCross Section Data:")
for k, v in xs.items():
    print(f"{k}: {v}")


## test 2
# Define your query parameters
burnup = 9.0
tf = 1242.0
tm = 557.0
bor = 400.0
group = 1.0

#################     real Cross Section Data:
#################     ABSORPTION: 0.00994697
#################     CAPTURE: 0.00794822
#################     FISSION: 0.00199874
#################     NU-FISSION: 0.00527257
#################     TRANSPORT: 0.237739
#################     OUT-SCATTER: 0.0171328
#################     DIFF(1/3TR): 1.4021

#,,0.00199874,0.00527257,0.237739,0.0171328,1.4021

# Call the get function (no try/except)
xs = vocab.get(burnup, tf, tm, bor, group)

# Print cross section data
print("\nCross Section Data:")
#for k, v in xs.items():
#    print(f"{k}: {v}")

##to create variables
ABSORPTION = xs["ABSORPTION"]
CAPTURE = xs["CAPTURE"]
FISSION = xs["FISSION"]
NU_FISSION = xs["NU-FISSION"]
TRANSPORT = xs["TRANSPORT"]
OUT_SCATTER = xs["OUT-SCATTER"]
DIFF = xs["DIFF(1/3TR)"]

print("ABSORPTION:", ABSORPTION)
print("CAPTURE:", CAPTURE)
print("FISSION:", FISSION)
print("NU_FISSION:", NU_FISSION)
print("TRANSPORT:", TRANSPORT)
print("OUT_SCATTER:", OUT_SCATTER)
print("DIFF:", DIFF)

"""