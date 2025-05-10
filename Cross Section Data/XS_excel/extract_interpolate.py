import pandas as pd

class CrossSectionVocabulary:
    def __init__(self, excel_file):
        # Load Excel file directly
        self.df = pd.read_excel(excel_file)

        # Build vocabulary as dictionary on the fly
        self.vocab = {}
        for _, row in self.df.iterrows():
            # Include 'ASSEMBLY' as part of the key
            key = (row["BURNUP"], row["TF"], row["TM"], row["BOR"], row["GROUP"], row["ASSEMBLY"])
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

    def get(self, burnup, tf, tm, bor, group, assembly):
        key = (burnup, tf, tm, bor, group, assembly)
        if key in self.vocab:
            return self.vocab[key]
        else:
            raise KeyError(f"No cross section data found for: BURNUP={burnup}, TF={tf}, TM={tm}, BOR={bor}, GROUP={group}, ASSEMBLY={assembly}")

    def list_keys(self):
        return list(self.vocab.keys())

# Function to handle both exact match and interpolation
def get_or_interpolate_cross_section(vocab, burnup, tf, tm, bor, group, assembly):
    # Filter the data for the fixed parameters (burnup, tf, tm, group, assembly)
    filtered_df = vocab.df[
        (vocab.df['BURNUP'] == burnup) &
        (vocab.df['TF'] == tf) &
        (vocab.df['TM'] == tm) &
        (vocab.df['GROUP'] == group) &
        (vocab.df['ASSEMBLY'] == assembly)
    ]
    
    # Check if the exact BOR value exists
    exact_match = filtered_df[filtered_df['BOR'] == bor]
    if not exact_match.empty:
        # If exact match found, return the data
        return exact_match.iloc[0].to_dict()
    
    # If exact match not found, interpolate
    sorted_df = filtered_df.sort_values(by='BOR')
    
    # Find the closest BOR values
    lower_bor = sorted_df[sorted_df['BOR'] <= bor].iloc[-1] if not sorted_df[sorted_df['BOR'] <= bor].empty else None
    upper_bor = sorted_df[sorted_df['BOR'] >= bor].iloc[0] if not sorted_df[sorted_df['BOR'] >= bor].empty else None

    if lower_bor is None or upper_bor is None:
        raise ValueError(f"Can't interpolate for BOR={bor}, it is out of the range.")

    # Perform linear interpolation between the two closest BOR values
    def interpolate_value(lower_value, upper_value, lower_bor, upper_bor, bor):
        return lower_value + (upper_value - lower_value) * (bor - lower_bor) / (upper_bor - lower_bor)

    # Interpolate for all cross-section data
    interpolated_data = {}
    for column in ['ABSORPTION', 'CAPTURE', 'FISSION', 'NU-FISSION', 'TRANSPORT', 'OUT-SCATTER', 'DIFF(1/3TR)']:
        interpolated_data[column] = interpolate_value(
            lower_bor[column], upper_bor[column], lower_bor['BOR'], upper_bor['BOR'], bor
        )
    
    return interpolated_data

# Example usage with external definitions
burnup = 0           # Define BURNUP externally
tf = 1019.3          # Define TF externally
tm = 557.0           # Define TM externally
group = 1            # Define GROUP externally
assembly = 1         # Define ASSEMBLY externally
test_bor = 1   # Define BOR value externally

vocab = CrossSectionVocabulary("XS_final.xlsx")

# Get or interpolate the data for BOR=100 with the external parameters
interpolated_bor_100 = get_or_interpolate_cross_section(vocab, burnup=burnup, tf=tf, tm=tm, bor=test_bor, group=group, assembly=assembly)

# Print interpolated data
print("Interpolated Data for BOR=1:")
for k, v in interpolated_bor_100.items():
    print(f"{k}: {v}")
