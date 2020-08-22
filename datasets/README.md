This folder includes a function to load 18 datasets for classification from the UCI Machine Learning repository 
(see http://archive.ics.uci.edu/ml/index.php). Each dataset is returned cleaned and stored in a dictionary 
with keys: _data_ (data matrices samples X features), _target_ (array with integer labels for different classes), and
_description_ with a short description of the data. Further information on each dataset can be found online.

Datasets available can be used to test new models as toy datasets or for benchmark comparisons and are saved
in a named tuple with fields (original name):

- biodeg: QSAR biodegradation;
- breastwi: Breast Cancer Wisconsin (Original);
- climate: Climate MOdel Simulation Crashes;
- banknote: banknote autentication; 
- ecoli: Ecoli
- glass: Glass identification; 
- ionosphere: Ionosphere;
- iris: Iris
- liver: Liver Disorders;
- movement: Libras Movement; 
- parkinsons: Parkinsons;
- seeds: seeds;
- transfusion: Blood Transfusion Service Center;
- wholesale: Wholesale customers;
- yeast: Yeast;
- forest: Forest type mapping;
- urban: Urban Land Cover;
- leaf: Leaf.

This code requires 

    Python>3.6

To run the code:
    
    from datasets.manuscript_builddatasets import build_ucidatasets

Then:

    uci_datasets = build_ucidatasets()
     
To access dataset field names:

    datasets._fields

To work with a specific dataset (e.g., yeast) run:

    datasets.yeast
