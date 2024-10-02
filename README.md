# Value of Information (VOI) Streamlit App 

## Installation

Install the package and dependencies:

```
pip install -e .
```

## Running Locally

```
streamlit run app.py --server.enableXsrfProtection false 
```

## Value of Information parameters
Users are able to model a "drill or walk away" decision and input prior probability of geothermal resource existing. Prior Value and Value of Perfect Information are ouput.

Next, the value of imperfect information is calculated, using a dataframe from various geophysical and geologicial observations around known geothermals systems and other locations deemed not a geothermal resource ("negative").

## Format of .csv files

You must upload one .csv file with data assosciated with positive label and one file with data assosciated with negative labels. **Both** must contain at a minimum a column labeled for the distance to the label: "PosSite_Distance" and "NegSite_Distance" resepectively.

Examples are shown [here](https://github.com/wtrainor/INGENIOUS_streamlit/tree/main/File%20Template)