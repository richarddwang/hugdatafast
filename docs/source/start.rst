==================
Get Started
==================

-----------------
Base use case
-----------------

.. code-block::

    >>> from nlp import load_dataset
    >>> from hugdatafast import *

.. note::
   This will also implicitly do ``from fastai.text.all import *``

Can you turn your data pipeline into only 3 lines ?

.. code-block::

    >>> dataset = load_dataset('glue', 'cola') 
    -> {'train': nlp.Dataset, 'validation': nlp.Dataset, 'test': nlp.Dataset}
    >>> tokenized_dataset = HF_TokenizeTfm(dataset, {'sentence':'text_idxs'}, hf_tokenizer).map() 
    >>> dls = HF_Datasets(tokenized_dataset, cols=['text_idxs', 'label'], hf_toker=hf_tokenizer).dataloaders(bs=64) 

Now you can enjoy 

1. :func:`show_batch` of fastai \n
Even you don't use fastai to train, you can still use as a normal DataLoader

.. code-block::

    >>> dls.show_batch(max_n=2)
                                                                                                                text_idxs       label
    --------------------------------------------------------------------------------------------------------------------------------------
    0  everybody who has ever , worked in any office which contained any type ##writer which had ever been used to type any      1
       letters which had to be signed by any administrator who ever worked in any department like mine will know what i mean .
    --------------------------------------------------------------------------------------------------------------------------------------
    1  playing with matches is ; lots of fun , but doing , so and empty ##ing gasoline from one can to another at the same       1
       time is a sport best reserved for arson ##s . [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]

2. Train model on the data using fastai, and also show the prediction

.. code-block::

    >>> learn = Learner(dls, your_model, loss_func=CrossEntropyLossFlat())
    >>> learn.fit(3)
    >>> learn.show_results()
                                                                                  text_idxs     label label_
    -----------------------------------------------------------------------------------------------------
    0	[CLS] scientists at the south hanoi institute of technology have succeeded in raising   1	   1 
      one dog with five legs , another with a cow ' s liver , and a third with no head . [SEP]	
    -----------------------------------------------------------------------------------------------------
    1 [CLS] as a teacher , you have to deal simultaneously with the administration ' s pressure   0    1
      on you to succeed , and the children ' s to be a nice guy . [SEP] [PAD] [PAD]
    
3. Use it as normal Dataloaders if you don't use fastai .

::

    >>> train_dataloader, val_dataloader, test_dataloader = dls[0], dls[1], dls[2]
    >>> for b in train_dataloader: break

------------------
Other use cases
------------------

1. Use your own dataset ?

* `nlp.Dataset s from local structured files (csv, json, ...) <https://huggingface.co/nlp/loading_datasets.html#from-local-files>`_

* `nlp.Dataset s from custom loading script <https://huggingface.co/nlp/add_dataset.html>`_

2. Use custom tokenization or custom processing function ?
use :class:`HF_Transform`

.. code-block::

    >>> def custom_tokenize(example):
    ...   example['tok_ids'] = hf_tokenizer.encode(example['sentence1'], example['sentence2'])
    ...   return example
    >>> tokenized_rte = HF_Transform(rte, custom_tokenize).map()

----------------------------
``hugdatafast`` in practice
----------------------------

You can see how to use ``hugdatafast`` in the real situations. Also, we're welcome you to share how you use 
``hugdatafast`` in your project, contact me via github or twitter to put your project link here.

* `electra_pytorch <https://github.com/richarddwang/hugdatafast>`_ : Pretrain ELECTRA and finetune on GLUE benchmark