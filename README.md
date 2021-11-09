<img src="https://github.com/Crisp-Unimib/ContrXT/blob/master/contrxt_logo.jpeg" alt="drawing" width="150"/>

# ContrXT
***A global, model-agnostic, contrastive explainer for any text classifier***

## Why do we might need ContrXT?
Imagine we have a text classifier, let's say M1, that has been retrained with new data generating M2.

1. Can we estimate to what extent the new model M2 classifies new data coherently to the past predictions made by the previous model M1?
2. Why does the criteria used by M1 result in class _c_, but M2 does not use the same criteria to classify as _c_ anymore?
3. Can we use natural language to explain the differences between models making them more comprehensible to final users?

Details on how ContrXT works can be found in this paper (bibtex here for citation). Here is a link to the video below.

[![Everything Is AWESOME](https://img.youtube.com/vi/pwQdinaXmDI/0.jpg)](https://www.youtube.com/watch?v=pwQdinaXmDI "Everything Is AWESOME")

## What ContrXT can do?

ContrXT is about **explaining how a classifier changed its predictions through time.** Alternatively, **it can be used to explain the differences in the classification behaviours of two distinct classifiers at a time.**

ContrXT takes as input the prediction of two distinct classifiers M1 and M2. Then, it traces the decision criteria of both classifiers by encoding the changes in the decision logic through Binary Decision Diagrams. Then (ii) it provides ``global, model-agnostic, time-contrastive (T-contrast)`` explanations in natural language, estimating why -and to what extent- the model has modified its behaviour over time.

## What ContrXT needs as input?
ContrXT takes as input the ``training data`` and ``the labels`` predicted by the classifier. This means you don't need to wrap ContrXT within your code at all!
As optional parameters, the user can specify:
- the coverage of the dataset to be used (default is 100%); otherwise a sampling procedure is used;
- to obtain explanations either for the multiclass case (default: one class vs all) or the two-class case (class vs class, by restricting the surrogate generation  to those classes);

## What ContrXT provides as output?
ContrXT provides two outputs:
### (1) Indicators to estimate which classes are changing more!
A picture estimating the differences among the classification criteria of both classifiers M1 and M2, that are **Add** and **Del** values. To estimate the degree of changes *among classes*, ContrXT also provides **add_global** and ***del_global***. In the case of a multiclass classifier,  ContrXT suggests focusing on classes that went through major alterations from M1 to M2, distinguishing between three groups according to their Add and Del values being above or below the 75th percentile.
![](https://github.com/Crisp-Unimib/ContrXT/blob/master/img/Add_Del_Magnitude_20N.png)
The picture above is geneated by ContrXT automatically. It provides indicators of changes in classification criteria from model M1 to M2 for each _20newsgroup_ class, using a DT as surrogate (0.8 fidelity) to explain two BERT classifiers.
### (2) Natural Language Explanations for each class
The indicators allow the user concentrating on classes that have changed more. For example, one might closely look at class _atheism_ as for this class the number of deleted paths is higher than the added ones
![](https://github.com/Crisp-Unimib/ContrXT/blob/master/img/alt.atheism.png)

The Natural Language Explanation for _atheism_ reveals the presence of the word _bill_ leads the retrained classifier M2 to assign the label _atheism_ to a specific record, whilst the presence of such a feature was not a criterion for the previous classifier M1.
Conversely, the explanation shows that M1 used the feature _keith_ to assign the label, whilst M2 discarded this rule. 

Actually, both terms refer to the name of the posts' authors: _Bill_'s posts are only contained within the dataset used to retrain whilst _Keith_'s ones are more frequent in initial dataset rather than the second one (dataset taken from _Jin, P., Zhang, Y., Chen, X., & Xia, Y. Bag-of-embeddings for text classification. In IJCAI-2016_).

Finally, M2 discarded the rule _having political atheist_ that was sufficient for M1 for classifying the instance.

## Do explanations lead to new insights?
Our experiments reveal Add/Del is not correlated with the models' accuracy.
For example, we assessed the presence of a correlation between _Add/Del_ and the change in performance of the classifiers in terms of F1-score on _20newsgroup_. To this end, we computed the Spearman's \rho between the _Add_ of every class and its change in f1-score between the two classifiers. 
The correlation values are not significant, p=0.91$, ro=-0.11 for the _Add_ and p=0.65, ro=-0.04 for _Del_ indicator. This confirms that Add/Del are not related to the f1-score of the a trained model. Instead, they estimate its behaviour change handling new data, considering which rules have been added or deleted with respect to the past.

## Limitation
At the moment, we support explaining predictions for text classifiers only, but we are working to extend it to deal with tabular data as well.

## Installation

In order to install ContrXT, clone this repository and then run:

```
pip install .
```

The PyEDA package is required but has not been added to the dependencies.
This is due to installation errors on Windows. If you are on Linux or Mac, you
should be able to install it by running:

```
pip3 install pyeda
```

However, if you are on Windows, we found that the best way to install is through
Christophe Gohlke’s [pythonlibs page](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyeda).
For further information, please consult the official PyEDA
[installation documentation](https://pyeda.readthedocs.io/en/latest/install.html).

In order to produce the PDF files, a Graphviz installation is also required.
Full documentation on how to install Graphviz on any platform is available
[here](https://graphviz.org/download/).

## Tutorials and Usage

A complete example of ContrXT usage is provided in the notebook "ContrXT Demo".
Complete documentation of the package functions will also be available shortly.

## Running tests

Basic unit tests are provided. In order to run them, after installation execute
the following command while in the main directory:

```
python -m unittest discover
```

## ContrXT as a Service through REST-API
ContrXT provides REST-API to generate explanations for any text classifier.  Our API enables users to get the outcome of ContrXT as discussed above (i.e., Indicators and Natural languange explanations) with no need to install or configure the tool locally. The required input from user are (i) the training data and (ii) the predicted labels by the classifier of their choice. 
Users are required to upload two csv files for two datasets for which the scema is shown in the following JSON.
```
schema = {
         "type" : "csv",
         "columns" : {
             "corpus" : {"type" : "string"},
             "category" : {"type" : "string"},
             "predicted" : {"type" : "string"},
         },
     }
```
The \contrxtv's API can be invoked using a few lines code shown in Code below.
```
code to call \contrxt API, label=code:API2]
import requests
import Zipfile as zp
import io
files = {
    'time_1': pd.read_csv('dt1.csv').to_csv().encode()
    'time_2': pd.read_csv('dt1.csv').to_csv().encode()
}
r = requests.post('[URLandPort]',files=files)
result = zp.ZipFile(io.BytesIO(r.content))
```
**Notice**. To avoid .... calls to our server, we ask users to ask for free credentials through this link. 