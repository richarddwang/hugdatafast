# Get Started

## Base use case

.. code-block::

    >>> from nlp import load_dataset
    >>> from hugdatafast import *

Can you turn your data pipeline into only 3 lines ?

.. code-block::

    >>> dataset = load_dataset('glue', 'cola') 
    -> {'train': nlp.Dataset, 'validation': nlp.Dataset, 'test': nlp.Dataset}
    >>> tokenized_dataset = HF_TokenizeTfm(dataset, {'sentence':'text_idxs'}, hf_tokenizer).map() 
    >>> dls = HF_Datasets(tokenized_dataset, cols=['text_idxs', 'label'], hf_toker=hf_tokenizer).dataloaders(bs=64) 

Now you can enjoy :func:`show_batch` and :func:`show_result` by :module:`fastai2`.\n
Even you don't use :module:`fastai2` to train, you can still use as a normal DataLoader
.. code-block::

    >>> dls.show_batch(max_n=2)
                                                                                                                text_idxs       label
    --------------------------------------------------------------------------------------------------------------------------------------
    0  everybody who has ever , worked in any office which contained any type ##writer which had ever been used to type any      1
       letters which had to be signed by any administrator who ever worked in any department like mine will know what i mean .
    --------------------------------------------------------------------------------------------------------------------------------------
    1  playing with matches is ; lots of fun , but doing , so and empty ##ing gasoline from one can to another at the same       1
       time is a sport best reserved for arson ##s . [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
    # use it as normal Dataloader you are familiar with if you don't use fastai2 ( try it !)
    >>> train_dataloader, val_dataloader, test_dataloader = dls[0], dls[1], dls[2]
    >>> for b in train_dataloader: break 

## Other use cases

