from pathlib import Path
import pyarrow as pa
import nlp
from fastai.text.all import *

class HF_BaseTransform():
  "The base class of HuggingFace/nlp transform. Inherit it to get the ability to do :func:`map` on not only a :class:`nlp.Dataset` but also  :class:`nlp.Dataset` s at once."

  def __init__(self, hf_dsets, remove_original=False, out_cols=None):
    """
    Args:
      hf_dsets (Dict[ :class:`nlp.Dataset` ] or :class:`nlp.Dataset` ): The Hugging Face dataset(s) to do `map`
      remove_original (bool): whther to remove all original columns after `map`
      out_cols (List[str]): output column names. If specified, it will assure they are not in the columns to be removed.
    """
    # check arguments
    if isinstance(hf_dsets, nlp.Dataset): hf_dsets = {'Single': hf_dsets}; self.single = True
    else: self.single = False
    assert isinstance(hf_dsets, dict)
    # save attributes
    self.hf_dsets = hf_dsets
    self.remove_original,self.out_cols = remove_original,out_cols

  @property
  def cache_dir(self): return Path(next(iter(self.hf_dsets.values())).cache_files[0]['filename']).parent

  @delegates(nlp.Dataset.map, but=["cache_file_name"])
  def map(self, split_kwargs=None, cache_dir=None, cache_name=None, **kwargs):
    """
    Args:
      split_kwargs (Dict[dict] or List[dict]): arguments of :func:`_map` and :func:`nlp.Dataset.map` for specific splits.
      cache_dir (str, default=`None`): if ``None``, it is the cache dir of the (first) dataset. 
      cache_name (str, default=`None`): format string includes one param "{split}" that will be converted to split name, as cache file name under `cache_dir` for each split. If ``None``, use automatically generated hashed name by hf/nlp.  
      kwargs: passed to :func:`_map` and :func:`nlp.Dataset.map`   
    """
    # check/process arguments
    if self.remove_original: 
      assert 'remove_columns' not in kwargs, "You have specified to remove all original columns."
    if split_kwargs is None:
      split_kwargs = { split:{} for split in self.hf_dsets }
    elif isinstance(split_kwargs, list):
      split_kwargs = { split:split_kwargs[i] for i, split in enumerate(self.hf_dsets) }
    elif isinstance(split_kwargs, dict):
      for split in split_kwargs.keys(): assert split in self.hf_dsets, f"{split} is not the split names {list(self.hf_dsets.keys())}."
      for split in self.hf_dsets:
        if split not in split_kwargs: split_kwargs[split] = {}
    cache_dir = Path(cache_dir) if cache_dir else self.cache_dir
    cache_dir.mkdir(exist_ok=True)
    # map
    new_dsets = {}
    for split, dset in self.hf_dsets.items():
      if self.remove_original: kwargs['remove_columns'] = dset.column_names
      if cache_name: 
        if self.single: kwargs['cache_file_name'] = str(cache_dir/cache_name)
        else: kwargs['cache_file_name'] = str(cache_dir/cache_name.format(split=split))
      kwargs.update(split_kwargs[split])
      if hasattr(kwargs, 'remove_columns'): self._check_outcols(kwargs['remove_columns'], split)
      new_dsets[split] = self._map(dset, split, **kwargs)
    # return
    if self.single: return new_dsets['Single']
    else: return new_dsets

  def _check_outcols(self, out_cols, rm_cols, split):
    if not self.out_cols: return
    for col in self.out_cols: assert col not in rm_cols, f"Output column name {col} is in the list of columns {rm_cols} will be removed after `map`." + f"The split is {split}" if split != 'Single' else ''

  @delegates(nlp.Dataset.map)
  def _map(self, dset, split, **kwargs):
    """ Child class can override this method to implement. """
    return dset.map(self, **kwargs)
  
  # The method you need to implement
  def __call__(self, example): 
    """ Child class should implement this method, which takes an example (dict) and return an example (dict), just like ``function`` in :func:`nlp.Dataset.map`"""
    raise NotImplementedError

@delegates()
class HF_Transform(HF_BaseTransform):
  """ Like normal :func:`nlp.Dataset.map`, but is able to process multiple :class:`nlp.Dataset` """
  def __init__(self, hf_dset, func, **kwargs):
    """
    Args:
      hf_dset (:class:`nlp.Dataset` or Dict[ :class:`nlp.Dataset` ]): The Hugging Face dataset(s) to map
      func (`Callable[dict]->dict`): like `func` in `nlp.Dataset.map`
    
    Example:
      >>> rte = nlp.load_dataset('glue', 'rte')
      >>> def custom_tokenize(example):
      ...   example['tok_ids'] = hf_tokenizer.encode(example['sentence1'], example['sentence2'])
      ...   return example
      >>> tokenized_rte = HF_Transform(rte, custom_tokenize).map()
      >>> tokenized_rte['validation'][0]
      {'sentence1': 'Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to the Christopher Reeve Foundation.',
       'sentence2': 'Christopher Reeve had an accident.',
       'label': 1,
       'idx': 0,
       'tok_ids': [101, 11271, 20726, 1010, 1996, 7794, 1997, 1996, 3364, 5696, 20726, 1010, 2038, 2351, 1997, 11192, 4456, 2012, 2287, 4008, 1010, 2429, 2000, 1996, 5696, 20726, 3192, 1012, 102, 5696, 20726, 2018, 2019, 4926, 1012, 102]}
    """
    super().__init__(hf_dset, **kwargs)
    self.func = func
  def __call__(self, example): return self.func(example)

@delegates(but=["out_cols"])
class HF_TokenizeTfm(HF_BaseTransform):
  
  def __init__(self, hf_dset, cols, hf_toker, **kwargs):
    """
    Args:
      hf_dset (:class:`nlp.Dataset` or Dict[ :class:`nlp.Dataset` ]): The Hugging Face dataset(s) to do tokenization
      cols: with one of the following signature:\n
        - `cols`(`Dict[str]`): tokenize the every column named [key] into a column named [value]
        - `cols`(`List[str]`): specify the name of columns to be tokenized, replace the original columns' data with tokenized one
      hf_toker (:class:`transformers.PreTrainedTokenizer`, optional): Hugging Face tokenizer

    Example:
      >>> cola = nlp.load_dataset('glue', 'cola') 
      {'train':nlp.Dataset, 'validation':nlp.Dataset, 'test':nlp.Dataset}
      >>> cola['train][0]
      {'sentence': "Our friends won't buy this analysis, let alone the next one we propose.",
        'label': 1, 'idx': 0,}
      >>> tokenized_cola = HF_TokenizeTfm(cola, {'sentence':'text_idxs'}, hf_tokenizer).map()
      >>> tokenized_cola['train'][0]
      {'sentence': "Our friends won't buy this analysis, let alone the next one we propose.",
        'label': 1, 'idx': 0,
        'text_idxs': [2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012]}
    """
    if isinstance(cols, list): cols = {c:c for c in cols}
    assert isinstance(cols, dict)
    super().__init__(hf_dset, out_cols=list(cols.values()), **kwargs)
    self.cols, self.tokenizer = cols, hf_toker
    """
    If don't specify cache file name, it will be hashed binary of pickled function that
    passed to `map`, so if you pass the same function, it knows to use cache.
    But tokenizer can't be pickled, so use tokenizer config to make tfms use different 
    tokenizers unique.  
    """
    self.tokenizer_config = hf_toker.pretrained_init_configuration
  
  def __call__(self, example):
    for in_col, out_col in self.cols.items():
      example[out_col] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(example[in_col]))
    return example

  def __getstate__(self):
    "specify something you don't want pickle here, remember to use copy to not modfiy orginal instance"
    state = self.__dict__.copy() 
    state['tokenizer'] = None 
    return state

class CombineTransform(HF_BaseTransform):
  """
  Base Class for Transform that combine multiple original samples into a new sample. 
  """
  def __init__(self, hf_dset, inp_cols, out_cols, init_attrs, drop_last=False):
    """
    Args:
      hf_dset (:class:`nlp.Dataset` or Dict[ :class:`nlp.Dataset` ]): The Hugging Face dataset(s) to do the transformation
      inp_cols (`List[str]`): names of input columns that used to produce samples
      out_cols (`List[str]`): names of output columns to put combined samples.
      init_attrs` (`List[str]`): name of attributes of children class that need to be their initial status when starts to aggregate dataset. i.e. Those defined in `__init__` and the value will changed during `accumulate`
      drop_last` (`Optional[bool]`, default: `False`): whether to drop the last accumulated sample.
    """
    super().__init__(hf_dset)
    self.inp_cols, self.out_cols =  inp_cols, out_cols
    for col in out_cols: assert col not in self.inp_cols, f"New column name can't be the same with any original column name. '{col}'"
    # batched map need dataset be in python format
    if isinstance(hf_dset, dict):
      for dset in hf_dset.values(): dset.set_format(type=None, columns=inp_cols) 
    else: hf_dset.set_format(type=None, columns=inp_cols)
    # dealing with last sample
    self.last_idx = len(hf_dset) - 1
    self.drop_last = drop_last
    # for reset
    self.init_attrs = init_attrs
    self.original_vals = [deepcopy(getattr(self, attr)) for attr in init_attrs]  

  def __call__(self, b, indices):
    # `nlp.Dataset.map` first test with several samples which affects our attrs, so we need to reinitialize.
    if 0 in indices: # reset
      for attr,val in zip(self.init_attrs, self.original_vals): setattr(self, attr, deepcopy(val))

    self.new_b = { c:[] for c in self.out_cols }
    for z in zip(*b.values()):
      self.accumulate(*z)
    
    # whehther put last example when it is last batch of `map`
    if not self.drop_last and self.last_idx in indices: 
      try: self.commit_example(self.create_example())
      except: pass # assume it is because there is nothing to create a example

    return self.new_b

  def commit_example(self, example):
    if example is None: return
    for col,val in example.items():
      self.new_b[col].append(val) 

  def accumulate(self, *args):
    """
    Child Class should implement this method.
    Given a example, do `self.commit_example(self.create_example()) when a new combined sample is ready.`
    Args:
      args : values of :data:`inp_cols` ( passed to :func:`__init__` ) of an example
    """
    raise NotImplementedError
  
  def create_example(self): 
    """
    Child Class should implement this method.
    Use attributes stored in the child class instance to create a combined sample (dict) when it is ready
    """
    raise NotImplementedError

  def _map(self, hf_dset, split, batch_size=1000, **kwargs):
    assert 'remove_columns' not in kwargs, "Aggregation type transform will only leave output columns for output dataset."
    return hf_dset.map(function=self, batched=True, batch_size=batch_size, with_indices=True, remove_columns=self.inp_cols, **kwargs)


@delegates(CombineTransform, but=["inp_cols", "out_cols", "init_attrs"])
class LMTransform(CombineTransform):
  """ Transform any dataset has tokenized text into dataset (autotgressive) language model. """
  def __init__(self, tokenized_hf_dset, max_len, text_col, x_text_col='x_text', y_text_col='y_text', **kwargs):
    """
    Args:
      tokenized_hf_dset (:class:`nlp.Dataset` or Dict[ :class:`nlp.Dataset` ]): tokenized Hugging Face dataset(s) to do LM transform 
      max_len (int): the length of a sentence
      text_col (str): the name of column that contains tokenized text (ids) of tokenized_hf_dset
      x_text_col (str): the name of the output column
      y_text_col (str): the name fo the output column
      kwargs: passed to :class:CombineTransform

    Example:
      >>> lm_dataset = LMTransform(tokenized_cola['validation'], max_len=20, text_col='text_idxs').map()
      >>> lm_dataset[0]
      {'x_text': [ 1996, 11279,  8469,  1996,  9478,  3154,  1997,  1996,  5749,  1012,
          1996, 15871,  2081,  1996,  8164,  7683,  2058,  1996,  4139,  3240],
       'y_text': [11279,  8469,  1996,  9478,  3154,  1997,  1996,  5749,  1012,  1996,
         15871,  2081,  1996,  8164,  7683,  2058,  1996,  4139,  3240,  1012]}
    """
    if isinstance(text_col, str): text_col = {text_col:['x_text','y_text']}
    assert isinstance(text_col, dict)
    self.text_col, (self.x_text_col, self.y_text_col) = next(iter(text_col.items()))
    self._max_len = max_len + 1
    self.residual_len, self.new_text = self._max_len, []
    super().__init__(tokenized_hf_dset, inp_cols=[self.text_col], out_cols=[x_text_col, y_text_col], init_attrs=['residual_len', 'new_text'], **kwargs)
    

  def accumulate(self, text): # *inp_cols
    """ Implement the abstract method"""
    usable_len = len(text)
    cursor = 0
    while usable_len != 0:
      use_len = min(usable_len, self.residual_len)
      self.new_text += text[cursor:cursor+use_len]
      self.residual_len -= use_len
      usable_len -= use_len
      cursor += use_len
      if self.residual_len == 0:
        self.commit_example(self.create_example())

  def create_example(self):
    """ Implement the abstract method"""
    # when read all data, the accumulated new_text might be less than two characters.
    if len(self.new_text) >= 2: 
      example = {self.x_text_col:self.new_text[:-1], self.y_text_col:self.new_text[1:]}
    else:
      example = None # mark "don't commit this"
    # reset accumulators
    self.new_text = []
    self.residual_len = self._max_len

    return example

@delegates(CombineTransform, but=["inp_cols", "out_cols", "init_attrs"])
class ELECTRADataTransform(CombineTransform):
  "Process any text corpus for ELECTRA's use"
  def __init__(self, hf_dset, is_docs, text_col, max_length, hf_toker, delimiter='\n', **kwargs):
    """
    Args:
      hf_dset (:class:`nlp.Dataset` or Dict[:class:`nlp.Dataset`]): **untokenized** Hugging Face dataset(s) to do the transform
      is_docs (bool): Whether each sample of this dataset is a doc
      text_col (str): the name of column of the dataset contains text 
      max_length (str): max length of each sentence
      hf_toker (:class:`transformers.PreTrainedTokenizer`): Hugging Face tokenizer
      delimiter (str): what is the delimiter to segment sentences in the input text
      kwargs: passed to :class:`CombineTransform`
    """
    self.is_docs = is_docs
    self.in_col = text_col
    self._current_sentences = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length
    self.cls_idx, self.sep_idx = hf_toker.cls_token_id, hf_toker.sep_token_id
    self.hf_toker = hf_toker
    self.delimiter = delimiter
    super().__init__(hf_dset, inp_cols=[self.in_col], out_cols=['input_ids','sentA_lenth'], 
                    init_attrs=['_current_sentences', '_current_length', '_target_length'], **kwargs)

  """
  This two main functions adapts official source code creates pretraining dataset, to CombineTransform
  """
  def accumulate(self, text):
    sentences = text.split(self.delimiter)
    for sentence in sentences:
      if not sentence: continue # skip empty
      tokids = self.hf_toker.convert_tokens_to_ids(self.hf_toker.tokenize(sentence))
      self.add_line(tokids)
    # end of doc
    if self.is_docs and self._current_length > 0:
      self.commit_example(self.create_example())
  
  def create_example(self):
    input_ids, sentA_lenth = self._create_example() # this line reset _current_sentences and _current_length in the end
    return {'input_ids': input_ids, 'sentA_lenth':sentA_lenth}
  # ...................................................

  def add_line(self, tokids):
    """Adds a line of text to the current example being built."""
    self._current_sentences.append(tokids)
    self._current_length += len(tokids)
    if self._current_length >= self._target_length:
      self.commit_example(self.create_example())

  def _create_example(self):
    """Creates a pre-training example from the current list of sentences."""
    # small chance to only have one segment as in classification tasks
    if random.random() < 0.1:
      first_segment_target_length = 100000
    else:
      # -3 due to not yet having [CLS]/[SEP] tokens in the input text
      first_segment_target_length = (self._target_length - 3) // 2

    first_segment = []
    second_segment = []
    for sentence in self._current_sentences:
      # the sentence goes to the first segment if (1) the first segment is
      # empty, (2) the sentence doesn't put the first segment over length or
      # (3) 50% of the time when it does put the first segment over length
      if (len(first_segment) == 0 or
          len(first_segment) + len(sentence) < first_segment_target_length or
          (len(second_segment) == 0 and
           len(first_segment) < first_segment_target_length and
           random.random() < 0.5)):
        first_segment += sentence
      else:
        second_segment += sentence

    # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
    first_segment = first_segment[:self._max_length - 2]
    second_segment = second_segment[:max(0, self._max_length -
                                         len(first_segment) - 3)]

    # prepare to start building the next example
    self._current_sentences = []
    self._current_length = 0
    # small chance for random-length instead of max_length-length example
    if random.random() < 0.05:
      self._target_length = random.randint(5, self._max_length)
    else:
      self._target_length = self._max_length

    return self._make_example(first_segment, second_segment)

  def _make_example(self, first_segment, second_segment):
    """Converts two "segments" of text into a tf.train.Example."""
    input_ids = [self.cls_idx] + first_segment + [self.sep_idx]
    sentA_lenth = len(input_ids)
    if second_segment:
      input_ids += second_segment + [self.sep_idx]
    return input_ids, sentA_lenth

  def __getstate__(self):
    "specify something you don't want pickle here, remember to use copy to not modfiy orginal instance"
    state = self.__dict__.copy() 
    state['hf_toker'] = None 
    return state