# ML-MDLText

ML-MDLText is a multilabel text classifier based on minimum description length principle. ML-MDLText is implemented as described in our paper:

> BITTENCOURT, MARCIELE M. ; SILVA, RENATO M. ; ALMEIDA, TIAGO A.,
> ML-MDLText: An efficient and lightweight multilabel text classifier 
> with incremental learning. APPLIED SOFT COMPUTING, v. 96, p. 1-15, 2020.


Additionally, a prototype version called ML-MDLText<sub>α</sub> used in experiments of paper presented at BRACIS'19 is also available in this repository. ML-MDLText<sub>α</sub>is implemented as described in our paper:

> BITTENCOURT, MARCIELE M. ; SILVA, RENATO M. ; ALMEIDA, TIAGO A.,
> ML-MDLText: A Multilabel Text Categorization Technique with 
> Incremental Learning, 2019 8th Brazilian Conference on Intelligent
> Systems (BRACIS), Salvador, Brazil, 2019, pp. 580-585.

These implementations is written in Python and built to work with [scikit-learn](https://scikit-learn.org/).

### How to use
To get these implementations, execute the following command:

`
$ git clone https://github.com/m-bittencourt/ML-MDLText.git
`

You can add the following code to your Python source code to guide the interpreter to find the package successfully:

```python
> import sys
> sys.path.append('src')
```
Import and initialize ML-MDLText or ML-MDLText<sub>α</sub> classifier with a multiclass classification method as meta-model (`clfClasses`):
```python
> from ML_MDLText_alpha import ML_MDLText_alpha
> classifier  = ML_MDLText_alpha(clfClasses)
```
```python
> from ML_MDLText import ML_MDLText
> classifier  = ML_MDLText(clfClasses)
```

Then, you can train the classifier with the training data (`x_train`) and their respective labels (` y_train`), according to the following code:
```python
> classifier.fit(x_train, y_train)
```
And test with test data (`x_test`) and the following code:
```python
> y_pred = classifier.predict(x_test)
```

### Running an example
In this repository, there is an example of using of ML-MDLText and ML-MDLText<sub>α</sub> with [medical database](http://mulan.sourceforge.net/datasets-mlc.html) and [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) as a meta-model, and can be executed through the following command:

`
$ python example.py
`

###Additional Information
If you find ML-MDLText helpful, please cite it as:

```
@INPROCEEDINGS{Bittencourt:2019,
	author={Marciele M. Bittencourt and Renato Moraes Silva and Tiago A. Almeida},
	title={{ML-MDLText}: A Multilabel Text Categorization Technique with Incremental Learning},
	booktitle={2019 8th Brazilian Conference on Intelligent Systems (BRACIS19)}, 
	year={2019},
	month=oct,
	address = {Salvador, BA, Brasil},
	publisher={IEEE},
	doi={10.1109/BRACIS.2019.00107},
	ISSN={2643-6256},
	number={},
	pages={580-585}
}
```

If you find ML-MDLText helpful, please cite it as:

```
@article{BITTENCOURT2020106699,
	title = {ML-MDLText: An efficient and lightweight multilabel text classifier 	with incremental learning},
	author = {Marciele M. Bittencourt and Renato M. Silva and Tiago A. Almeida},
	journal = {Applied Soft Computing},
	volume = {96},
	pages = {106699},
	year = {2020},
	issn = {1568-4946},
	doi = {https://doi.org/10.1016/j.asoc.2020.106699}
}
```
