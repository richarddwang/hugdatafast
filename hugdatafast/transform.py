from pathlib import Path
import pyarrow as pa
import datasets
from fastai.text.all import *

@patch
def cache_directory(self: datasets.arrow_dataset.Dataset):
  return os.path.abspath(os.path.dirname(self.cache_files[0]['filename']))

@patch
def my_map(self: datasets.arrow_dataset.Dataset, *args, **kwargs):
  """
  The same as :class:`datasets.arrow_dataset.Dataset` , but it can add cache directory and .arrow to cache_file_name autmomatically for us.
  
  Example:
    >>> dataset.map(a_func, cache_file_name='processed')
    # cache file path become "<dataset cache directory>/processed.arrow"
  """
  cache_file_name = kwargs.pop('cache_file_name', None)
  if cache_file_name is not None:
    if not cache_file_name.endswith('.arrow'): cache_file_name += '.arrow'
    if '/' not in cache_file_name: cache_file_name = os.path.join(self.cache_directory(), cache_file_name)
  return self.map(*args, cache_file_name=cache_file_name, **kwargs)

@patch
def my_map(self: datasets.dataset_dict.DatasetDict, *args, **kwargs):
  """
  The same as :class:`datasets.dataset_dict.DatasetDict` , but it can infer cache names for us.

  Example:
    >>> datasets.map(a_func, cache_file_names='processed_{split}')
    # cache file paths : "<dataset cache directory>/processed_train.arrow", "<dataset cache directory>/processed_validation.arrow", "<dataset cache directory>/processed_test.arrow"
  """
  # cache file names
  cache_file_names = kwargs.pop('cache_file_names', None)
  self._check_values_type()
  if cache_file_names is None: cache_file_names = {k: None for k in self}
  if isinstance(cache_file_names, str): cache_file_names = {k: cache_file_names.format(split=k) for k in self}
  # split specific kwargs
  fn_kwargs = kwargs.pop('fn_kwargs', None)
  if fn_kwargs is None: fn_kwargs = {}
  _fn_kwargs = {split_name:{} for split_name in self.keys()}
  for k,v in fn_kwargs.items():
    if k in _fn_kwargs and isinstance(v, dict): # kwargs for a specific split
      _fn_kwargs[k] = v
    else: # generic kwargs for all splits
      for split in _fn_kwargs: _fn_kwargs[split][k] = v

  # pass
  return datasets.dataset_dict.DatasetDict({k: dataset.my_map(*args, 
                                                              cache_file_name=cache_file_names[k], 
                                                              fn_kwargs=_fn_kwargs[k], 
                                                              **kwargs) for k, dataset in self.items()})

class SimpleTokenize():
  def __init__(self, cols, hf_toker):
    if isinstance(cols, list): cols = {c:c for c in cols}
    elif isinstance(cols, str): cols = {cols:cols}
    assert isinstance(cols, dict)
    self.cols = cols
    self.hf_toker = hf_toker
  def __call__(self, example):
    for in_col, out_col in self.cols.items():
      example[out_col] = self.hf_toker.convert_tokens_to_ids(self.hf_toker.tokenize(example[in_col]))
    return example

class CombineTransform():
  """
  Base Class for Transform that combine multiple original samples into a new sample. 
  """
  def __init__(self, hf_dset, in_cols, out_cols, drop_last=False):
    """
    Args:
      hf_dset (:class:`Dataset` or :class:`DatasetDict`): The Hugging Face dataset(s) to do the transformation
      in_cols (`List[str]`): names of input columns that used to produce samples
      out_cols (`List[str]`): names of output columns to put combined samples.
      drop_last` (`Optional[bool]`, default: `False`): whether to drop the last accumulated sample.
    """
    # Always do the case of multiple datasets for the convenience of coding
    if isinstance(hf_dset, datasets.arrow_dataset.Dataset): self.dsets = {'Single': hf_dset}; self.single=True
    else: self.dsets = hf_dset; self.single=False
    
    # check column names
    self.in_cols, self.out_cols =  in_cols, out_cols
    for col in out_cols: assert col not in self.in_cols, f"New column name can't be the same with any original column name. '{col}'"
    
    # batched map need dataset in Python format
    for dset in self.dsets.values(): dset.set_format(type=None, columns=in_cols)
    
    # dealing with last sample
    self.last_idx = len(hf_dset) - 1
    self.drop_last = drop_last 

  def __call__(self, b, indices):
    # If first batch, `datasets.Dataset.map` first test with several samples which affects our internal states, so we need to reinitialize.
    if 0 in indices:
      self.reset_states()

    self.new_b = { c:[] for c in self.out_cols }
    values = [ b[c] for c in self.in_cols ]
    for z in zip(*values):
      self.accumulate(*z)
    
    # If Last batch, whehther commit last incomplete example
    if not self.drop_last and self.last_idx in indices:
      try: self.commit_example(self.create_example())
      except: pass # assume it is because there's nothing can be created

    return self.new_b

  def commit_example(self, example):
    if example is None: return
    for col,val in example.items():
      self.new_b[col].append(val)

  def reset_states(self):
    """
    Child Class should implement this method.\n
    Reset all containers, flags to their initial values.
    """
    raise NotImplementedError

  def accumulate(self, *args):
    """
    Child Class should implement this method.\n
    Given a example, do `self.commit_example(self.create_example()) when a new combined sample is ready.`
    Args:
      args : values of :data:`inp_cols` ( passed to :func:`__init__` ) of an example
    """
    raise NotImplementedError
  
  def create_example(self): 
    """
    Child Class should implement this method.\n
    Use internal states stored in the child class instance to create a combined example (dict).\n
    When nothing can't be created, return ``None`` or raise any exception to show it.
    """
    raise NotImplementedError

  def map(self, batch_size=1000, cache_file_name=None, **kwargs):
    """
    Args:
      batch_size(int): See :class:`datasets.Dataset.map`, shouldn't be None here
      cache_file_name: The same with the one of :func:`my_map`
      kwargs: passed to :class:`datasets.Dataset.map`
    """

    # check
    assert 'remove_columns' not in kwargs, "Aggregation type transform will only leave output columns for output dataset."
    
    # infer cache_file_name s
    if not isinstance(cache_file_name, dict):
      cache_names = { k:cache_file_name for k in self.dsets.keys() }
    for k, dset in self.dsets.items():
      if cache_names[k] is None: continue
      if not cache_names[k].endswith('.arrow'): cache_names[k] += '.arrow'
      if '{split}' in cache_names[k]: cache_names[k] = cache_names[k].format(split=k)
      if '/' not in cache_names[k]: cache_names[k] = os.path.join(dset.cache_directory(), cache_names[k])
    
    # map every dataset
    mapped_dsets = {}
    for k, dset in self.dsets.items():
      self.last_idx = len(dset) - 1
      mapped_dset = dset.map(function=self, 
                             batched=True, batch_size=batch_size, 
                             with_indices=True,
                             num_proc=1,
                             cache_file_name=cache_names[k],
                             remove_columns=self.in_cols, # Cuz output column has less rows (combined) than orginal column
                             **kwargs)
      mapped_dset.set_format(None, columns=self.out_cols)
      mapped_dsets[k] = mapped_dset

    if self.single: return mapped_dsets['Single']
    else: return datasets.DatasetDict(mapped_dsets)

@delegates(CombineTransform, but=["inp_cols", "out_cols", "init_attrs"])
class LMTransform(CombineTransform):
  """ 
  Transform any dataset has tokenized text into dataset (autotgressive) language model.
  !! Caution: This span context window across examples. So make sure your texts in examples of the datasets are consecutive or relative.
  Or you are knowing what you are doing.
  """
  def __init__(self, tokenized_hf_dset, max_len, text_col, x_text_col='x_text', y_text_col='y_text', **kwargs):
    """
    Args:
      tokenized_hf_dset (:class:`Dataset` or :class:`DatasetDict`): tokenized Hugging Face dataset(s) to do LM transform 
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
    self.reset_states()
    super().__init__(tokenized_hf_dset, in_cols=[self.text_col], out_cols=[x_text_col, y_text_col], **kwargs)
    
  def reset_states(self):
    self.new_text = []
    self.residual_len = self._max_len

  def create_example(self):
    # when read all data, the accumulated new_text might be less than two characters.
    if len(self.new_text) >= 2: 
      example = {self.x_text_col:self.new_text[:-1], self.y_text_col:self.new_text[1:]}
    else:
      example = None # mark "don't commit this"
    # reset accumulators
    self.reset_states()

    return example

  def accumulate(self, text): # *inp_cols
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

@delegates(CombineTransform, but=["inp_cols", "out_cols", "init_attrs"])
class ELECTRADataTransform(CombineTransform):
  "Process any text corpus for ELECTRA's use"
  def __init__(self, hf_dset, is_docs, text_col, max_length, hf_toker, delimiter='\n', **kwargs):
    """
    Args:
      hf_dset (:class:`Dataset` or :class:`DatasetDict`): **untokenized** Hugging Face dataset(s) to do the transform
      is_docs (bool): Whether each sample of this dataset is a doc
      text_col (str): the name of column of the dataset contains text 
      max_length (str): max length of each sentence
      hf_toker (:class:`transformers.PreTrainedTokenizer`): Hugging Face tokenizer
      delimiter (str): what is the delimiter to segment sentences in the input text
      kwargs: passed to :class:`CombineTransform`
    """
    self.is_docs = is_docs
    self.in_col = text_col
    self._max_length = max_length
    self.cls_idx, self.sep_idx = hf_toker.cls_token_id, hf_toker.sep_token_id
    self.hf_toker = hf_toker
    self.delimiter = delimiter
    self.reset_states()
    super().__init__(hf_dset, in_cols=[self.in_col], out_cols=['input_ids','sentA_lenth'], **kwargs)

  """
  These three main functions adapts official source code creates pretraining dataset, to CombineTransform
  """
  def reset_states(self):
    self._current_sentences = []
    self._current_length = 0
    self._target_length = self._max_length

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
    ## small chance for random-length instead of max_length example
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