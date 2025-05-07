import re
import pandas as pd
import pickle

input_filename = "NUu250c00_Cross_Sections.txt"

# Prepare storage for data
data = []

# Regex patterns
header_pattern = re.compile(r"\*+\s*BURNUP\s*=\s*([\d\.Ee+-]+)\s*V=.*?TF\s*=\s*([\d\.Ee+-]+)\s*TM\s*=\s*([\d\.Ee+-]+)\s*BOR\s*=\s*([\d\.Ee+-]+)")
group_pattern = re.compile(r"^\s*(\d+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)")

current_burnup = None
current_tf = None
current_tm = None
current_bor = None

# Read file
with open(input_filename, "r") as f:
    for line in f:
        line = line.strip()

        header_match = header_pattern.search(line)
        if header_match:
            current_burnup = float(header_match.group(1))
            current_tf = float(header_match.group(2))
            current_tm = float(header_match.group(3))
            current_bor = float(header_match.group(4))
            continue

        group_match = group_pattern.search(line)
        if group_match:
            group_data = {
                "BURNUP": current_burnup,
                "TF": current_tf,
                "TM": current_tm,
                "BOR": current_bor,
                "GROUP": int(group_match.group(1)),
                "ABSORPTION": float(group_match.group(2)),
                "CAPTURE": float(group_match.group(3)),
                "FISSION": float(group_match.group(4)),
                "NU-FISSION": float(group_match.group(5)),
                "TRANSPORT": float(group_match.group(6)),
                "OUT-SCATTER": float(group_match.group(7)),
                "DIFF(1/3TR)": float(group_match.group(8)),
            }
            data.append(group_data)

# Build dataframe
df = pd.DataFrame(data)

# Save dataframe
df.to_csv("XS.csv", index=False)
print("Parsed data saved to NUu250c00_Cross_Sections_parsed.csv")
