Tabulate file just creates a csv file which is needed by extract:
The code defines a class CrossSectionVocabulary that loads cross section data from a CSV file and allows 
querying that data based on 5 parameters:

BURNUP

TF (Fuel Temperature)
TM (Moderator Temperature)
BOR (Boron Concentration)
GROUP (Energy Group)

It returns detailed cross section values (ABSORPTION, CAPTURE, etc.) for the queried parameters.
