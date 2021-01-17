# Run manuscript examples

Manuscript examples can be run via Python interpreter or within a Jupyter notebook.

## Interpreter
Open the file `manuscript_examples.py` and add to the `main` the examples 
to run (i.e., `example_1/2/3/4()`).

`cd` to the top-level directory, then in terminal run:

```
python -m working_examples.manuscript_examples
``` 

To save the output to a file write in `main` the name of the logging file then run the interpreter as above.

The output file is saved in the top-level project folder.

## Jupyter notebook

From the `working_examples` directory open a terminal session and 
call 

```
jupyter-lab
```

Then open the file `manuscript_examples.ipynb` and run the desired cells.

## Additional examples

To run `blobs.py`, `data_dimensionality.py`, `handwritten_digits.py`, `reval_bigocomplexity.py`, and 
`reval_timeitcomplexity.py`, from the top-level directory:

```
cd working_examples
python file_name.py
```
