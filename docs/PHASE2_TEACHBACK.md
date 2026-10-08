# Phase 2 Teach-back: Cheminformatics and Molecular Descriptor Generation

## What RDKit is and why we use it

RDKit is an open-source toolkit for cheminformatics. It provides Python objects and functions for working with molecules, reading chemical file formats, calculating structural properties, and searching chemical databases.

In this project, RDKit converts SMILES strings into molecular objects and calculates descriptors that can serve as numerical inputs for a machine-learning model. The descriptors capture chemically meaningful information without requiring an experimental measurement for every prediction.

## What SMILES strings represent

SMILES (Simplified Molecular Input Line Entry System) is a compact text representation of a molecule. Each character sequence describes atoms and their connections. For example, `CCO` represents ethanol: a two-carbon chain ending with an oxygen-bearing carbon.

A SMILES string is not a molecule by itself; it is a text encoding of one. RDKit parses that string into an internal molecular graph, where atoms are nodes and chemical bonds are edges. A valid structure must satisfy the rules of valence and bonding for the atoms represented.

## The six descriptors and their chemical significance

1. **MolWt (molecular weight)**
   - The sum of the atomic masses in the molecule.
   - It affects physical properties such as vapor pressure, diffusion, and solubility.

2. **LogP**
   - The calculated octanol/water partition coefficient.
   - A higher LogP generally indicates greater hydrophobic character and a greater tendency to partition into an organic phase.
   - LogP is calculated from molecular structure, not measured from the same experimental system as LogS.

3. **TPSA (topological polar surface area)**
   - The total surface area contributed by polar atoms and bonds.
   - It is correlated with hydrogen bonding and membrane permeability.

4. **HBD (hydrogen-bond donors)**
   - The number of heteroatoms or groups capable of donating a hydrogen bond.
   - A molecule with more donor groups may form stronger interactions with water or biological targets.

5. **HBA (hydrogen-bond acceptors)**
   - The number of atoms or groups capable of accepting a hydrogen bond.
   - This is often important for predicting solubility and binding.

6. **RotBonds (rotatable bonds)**
   - The number of bonds that can rotate without breaking a bond or changing the molecular framework.
   - More rotatable bonds often increase conformational flexibility and can affect binding and physical properties.

## How `calculate_descriptors()` works

The function accepts a single SMILES string and returns a dictionary containing the six descriptor values.

1. It checks that the input is a non-empty string.
2. It calls `Chem.MolFromSmiles(smiles)` to convert the string into an RDKit molecule.
3. If the string is invalid, `MolFromSmiles()` returns `None`, and the function returns `None`.
4. When the structure is valid, the function passes the molecule to RDKit descriptor functions.
5. It validates that each value is present and finite.
6. It returns a dictionary with keys exactly matching the required descriptor names.

The function is intentionally defensive: one invalid structure does not stop the entire dataset from being processed.

## How Pandas `.apply()`, DataFrames, and `pd.concat()` work in this pipeline

A **DataFrame** is a two-dimensional table with labeled rows and columns. The ESOL dataset is loaded as a DataFrame, with each row representing one molecule.

The `.apply()` method executes a function across DataFrame cells, rows, or columns. In this Phase 2 implementation, descriptor conversion is handled in a row-wise loop rather than a broad `apply()` call, which keeps the logic explicit and easier to debug.

A DataFrame may also be built from a list of row dictionaries. Each dictionary contains the compound identifier, the SMILES string, the experimental LogS value, and the six descriptors.

`pd.concat()` is useful when combining multiple DataFrames with compatible columns. It can append rows from several processed subsets or add one DataFrame of calculated features to a DataFrame containing other metadata. In a larger pipeline, relational operations or index alignment may be needed to ensure that rows remain correctly matched.

## The difference between LogP and LogS

LogP is a structural descriptor describing the favored phase of a neutral molecule under a defined octanol/water partition experiment. It describes the tendency of a molecule to prefer an organic environment over water.

LogS is the measured or modeled aqueous solubility. It is the base-10 logarithm of the solubility expressed in mols per litre. A more negative LogS value means lower solubility in water.

A molecule can have a high LogP and still be highly soluble if it has many ionizable or polar groups. In this project, LogP is calculated from structure and used as a molecular feature, while experimental LogS is the target variable.

## Why we use experimental LogS rather than the dataset's existing predictions

The ESOL dataset includes an experimentally measured LogS value and an existing predicted LogS value. The predicted value was produced by a model or an earlier prediction method, and it is not a direct experimental measurement.

For Phase 2, we should not use the prediction column as either a model feature or a target because:

- it would introduce leakage from a model-generated quantity;
- it would make the task less useful as a test of learning from molecular structure;
- it would blur the distinction between an observed outcome and an estimate of that outcome.

The experimental LogS is the chemically meaningful target for a supervised model. We retain only rows with valid experimental values and valid molecular structures.

## How invalid structures and missing data are handled

The processing function checks each row in this order:

1. The Compound ID is present.
2. The SMILES string is not missing or empty.
3. The experimental LogS is present and numeric.
4. RDKit can parse the structure.
5. The resulting descriptor values are numeric and finite.

Rows that fail these checks are excluded from the processed dataset. The pipeline reports the number of excluded records and the reason for each exclusion category.

This approach is conservative. It prevents invalid chemistry from being passed into a model and makes the dataset cleaning process transparent.

## Interview questions and answers

### 1. What is the difference between a chemical descriptor and a chemical feature?

A chemical descriptor is a calculated or measured numerical property derived from a molecular structure. In machine learning, a descriptor becomes a feature when it is used as an input variable to represent a molecule.

### 2. Why can a SMILES string be invalid even though it looks readable?

A SMILES string may contain a wrong valence, an impossible bond pattern, an unknown atom type, or an unbalanced charge. RDKit checks the chemical graph and rejects strings that cannot represent a valid molecule.

### 3. What does TPSA tell us about a molecule?

TPSA estimates the polar surface area of a molecule. It is often associated with hydrogen bonding and membrane permeability. A larger TPSA generally indicates greater polarity and often a lower tendency to cross certain biological membranes.

### 4. Why can LogP and LogS differ for the same molecule?

LogP describes the relative preference between octanol and water, while LogS describes the absolute amount dissolved in water. They are related but are not the same physical quantity.

### 5. Why is it important to exclude invalid molecules before model training?

Invalid molecules cannot provide reliable chemical information. Including them could produce undefined descriptors, incorrect features, or model failures. Removing invalid rows makes the training data consistent and makes the cleaning process easier to explain.

## Two-minute interview explanation

"In Phase 2, we build the chemical representation that will later feed a machine-learning model. We start with the ESOL dataset, where each row contains a molecule identified by a SMILES string and an experimentally measured aqueous solubility. RDKit parses each SMILES string into a molecular object, and we calculate six descriptors: molecular weight, LogP, TPSA, hydrogen-bond donors, hydrogen-bond acceptors, and rotatable bonds.

These descriptors summarize molecular structure in numeric form. We filter out missing values, malformed SMILES strings, non-numeric LogS values, and invalid structures. The resulting dataset contains the compound identifier, SMILES, experimental LogS, and the six descriptor columns. We then validate that the output contains no missing or infinite values and that every retained molecule can be parsed by RDKit.

The important modeling choice is that we use the measured LogS as the target and do not use the existing prediction column. This keeps the dataset scientifically grounded and avoids leakage from a model-generated value. The output of Phase 2 is a cleaned, validated feature matrix ready for the machine-learning steps in Phase 3."
