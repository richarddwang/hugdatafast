==================
Get Started
==================

-----------------
Base use case
-----------------

::

    >>> from datasets import load_dataset
    >>> from hugdatafast import *

.. note::
   This will also implicitly do ``from fastai.text.all import *``

Can you turn your data pipeline into only 3 lines ?

::

    >>> datasets = load_dataset('glue', 'cola') 
    -> {'train': datasets.Dataset, 'validation': datasets.Dataset, 'test': datasets.Dataset}
    >>> tokenized_datasets = datasets.map(simple_tokenize_func({'sentence':'text_idxs'}, hf_tokenizer))
    >>> dls = HF_Datasets(tokenized_datasets, cols=['text_idxs', 'label'], hf_toker=hf_tokenizer).dataloaders(bs=64) 

Now you can enjoy 

1. :func:`show_batch` of fastai \n
Inspect your processed data and quickly check if there is anything wrong with your data processing.

::

    >>> dls.show_batch(max_n=2)
                                                                                                                text_idxs       label
    --------------------------------------------------------------------------------------------------------------------------------------
    0  everybody who has ever , worked in any office which contained any type ##writer which had ever been used to type any      1
       letters which had to be signed by any administrator who ever worked in any department like mine will know what i mean .
    --------------------------------------------------------------------------------------------------------------------------------------
    1  playing with matches is ; lots of fun , but doing , so and empty ##ing gasoline from one can to another at the same       1
       time is a sport best reserved for arson ##s . [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]

2. Train model on the data using fastai, and also show the prediction

::

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

* `datasets.Dataset s from local structured files (csv, json, ...) <https://huggingface.co/datasets/loading_datasets.html#from-local-files>`_

* `datasets.Dataset s from custom loading script <https://huggingface.co/datasets/add_dataset.html>`_

2. Need to combine examples to generate new example ? (e.g. Traditional language model) 

::

    >>> lm_datasets = LMTransform(datasets, max_len=20, text_col='text_idxs').map()
    >>> hf_tokenizer.decode(lm_datasets['validation'][-1]['x_text'])
    . john talked to bill about himself
    >>> hf_tokenizer.decode(lm_datasets['validation'][-1]['y_text'])
    john talked to bill about himself.

If you want to implement your own logic to combine examples, try to extend :class:`CombineTransform`.

----------------------------
``hugdatafast`` in practice
----------------------------

You can see how to use ``hugdatafast`` in the real situations. Also, You are welcome to share how you use 
``hugdatafast`` in your project, contact me via github or twitter to put your project link here.

* `electra_pytorch <https://github.com/richarddwang/electra_pytorch>`_ : Pretrain ELECTRA and finetune on GLUE benchmark