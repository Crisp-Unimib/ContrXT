Welcome to ContrXT's documentation!
===================================

**ContrXT** is a Python library that provides **global, model-agnostic,
contrastive explainer for any text classifier**.

ContrXT is *about explaining how a classifier changed its predictions through time.
Also, it can be used to explain the differences in the classification behaviours
of two distinct classifiers at a time*.

ContrXT takes as input the predictions of two distinct classifiers M1 and M2.
Then, it traces the decision criteria of both classifiers by encoding the changes
in the decision logic through Binary Decision Diagrams. Then (ii) it provides
"global, model-agnostic, time-contrastive (T-contrast) "explanations in natural
language, estimating why -and to what extent- the model has modified its behaviour
over time.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   api
