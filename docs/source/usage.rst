Usage
=====

.. _installation:

Installation
------------

To use ContrXT, first install it using pip:

.. code-block:: console

   (.venv) $ pip install contrxt

The PyEDA package is required but has not been added to the dependencies.
This is due to installation errors on Windows.
If you are on Linux or Mac, you should be able to install it by running:

.. code-block:: console

   (.venv) $ pip install pyeda

However, if you are on Windows, we found that the best way to install is
through `Christophe Gohlke's pythonlibs page<https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyeda/>`_.

For further information, please consult the official PyEDA
`installation documentation<https://pyeda.readthedocs.io/en/latest/install.html>`_.

To produce the PDF files, a Graphviz installation is also required.
Full documentation on how to install Graphviz on any platform is available
`here<https://graphviz.org/download/>`_.

Using ContrXT
----------------

TO-DO documentation.
[...]you can use the ``exp.run_trace()`` function[...]

For example:

>>> from contrxt.contrxt import ContrXT
>>> exp = ContrXT(X_t1, predicted_labels_t1,
>>>               X_t2, predicted_labels_t2)
>>> exp.run_trace()
>>> exp.run_explain()
>>> exp.explain.BDD2Text()
