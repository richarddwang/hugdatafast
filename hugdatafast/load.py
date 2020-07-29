from functools import partial
from pathlib import Path
import json
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import nlp
from fastai2.text.all import *



@delegates()
class MySortedDL(TfmdDL):
    "A :class:`DataLoader` that do smart batching and dynamic padding. Different from :class:`SortedDL`, it automatically pad every attribute of samples, is able to filter samples, and can be cached to sort/filter only at first time."

    def __init__(self, dataset, srtkey_fc=None, filter_fc=False, pad_idx=None, cache_file=None, **kwargs):
        """
        Args:
            dataset (HF_Dataset): Actually any object implements ``__len__`` and ``__getitem__`` that return a tuple as a sample.
            srtkey_fc (``*args->int``, optional): Get elements of a sample and return a sorting key. 
              If ``None``, sort by length of first element of a sample.
              If ``False``, not sort. 
            filter_fc (``*args->bool``, optional): Get elements of a sample and return ``True`` to keep the sample.
            pad_idx (``int``, optional): pad each attribute of samples to the max length of its max length within the batch. 
              If ``False``, do no padding. 
              If ``None``, try ``dataset.pad_idx``, do no padding if no such attribute.
            cache_file (``str``, optional): Path of a json file to cache info for sorting and filtering.

        Examples::
            class Fake_HF_Dataset()
        """
        # Defaults
        if srtkey_fc is not False: srtkey_fc = lambda *x: len(x[0])
        if pad_idx is None: pad_idx = getattr(dataset, 'pad_idx', False)
        cache_file = Path(cache_file) if cache_file else None
        idmap = list(range(len(dataset)))

        # Save attributes
        super().__init__(dataset, **kwargs)
        store_attr(self, 'pad_idx,srtkey_fc,filter_fc,cache_file,idmap')

        # Prepare records for sorting / filtered samples
        if srtkey_fc or filter_fc:
          if cache_file and cache_file.exists():
            # load cache and check
            with cache_file.open(mode='r') as f: cache = json.load(f)
            idmap, srtkeys = cache['idmap'], cache['srtkeys']
            if srtkey_fc: 
              assert srtkeys, "srtkey_fc is passed, but it seems you didn't sort samples when creating cache."
              self.srtkeys = srtkeys
            if filter_fc:
              assert idmap, "filter_fc is passed, but it seems you didn't filter samples when creating cache."
              self.idmap = idmap
          else:
            # overwrite idmap if filter, get sorting keys if sort
            idmap = []; srtkeys = []
            for i in tqdm(range_of(dataset), leave=False):
                sample = self.do_item(i)
                if filter_fc and not filter_fc(*sample): continue
                if filter_fc: idmap.append(i)
                if srtkey_fc: srtkeys.append(srtkey_fc(*sample))
            if filter_fc: self.idmap = idmap
            if srtkey_fc: self.srtkeys = srtkeys
            # save to cache
            if cache_file:
              try: 
                with cache_file.open(mode='w+') as f: json.dump({'idmap':idmap,'srtkeys':srtkeys}, f)
              except: os.remove(str(cache_file))
          # an info for sorting
          if srtkey_fc: self.idx_max = np.argmax(self.srtkeys)
          # update number of samples
          if filter_fc: self.n = self.n = len(self.idmap)

    def create_item(self, i): return self.dataset[self.idmap[i]]

    def create_batch(self, samples):
        if self.pad_idx is False: return super().create_batch(samples)
        return tuple( pad_sequence(attr, batch_first=True, padding_value=self.pad_idx) if attr[0].shape else torch.stack(attr) for i, attr in enumerate(zip(*samples)))

    def get_idxs(self):
        idxs = super().get_idxs()
        if self.shuffle: return idxs
        if self.srtkey_fc: return sorted(idxs, key=lambda i: self.srtkeys[i], reverse=True)
        return idxs

    def shuffle_fn(self,idxs):
        if not self.srtkey_fc: return super().shuffle_fn(idxs)
        idxs = np.random.permutation(self.n)
        idx_max = np.where(idxs==self.idx_max)[0][0]
        idxs[0],idxs[idx_max] = idxs[idx_max],idxs[0]
        sz = self.bs*50
        chunks = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        chunks = [sorted(s, key=lambda i: self.srtkeys[i], reverse=True) for s in chunks]
        sort_idx = np.concatenate(chunks)

        sz = self.bs
        batches = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        sort_idx = np.concatenate(np.random.permutation(batches[1:-1])) if len(batches) > 2 else np.array([],dtype=np.int)
        sort_idx = np.concatenate((batches[0], sort_idx) if len(batches)==1 else (batches[0], sort_idx, batches[-1]))
        return iter(sort_idx)

    @delegates(TfmdDL.new)
    def new(self, dataset=None, **kwargs):
        # We don't use filter_fc here cuz we can't don't validate certaion samples in dev/test set. 
        return super().new(dataset=dataset, pad_idx=self.pad_idx, srtkey_fc=self.srtkey_fc, filter_fc=False, **kwargs)

"To pr"
# only change "label" to "title"
def my_show_title(o, ax=None, ctx=None, title=None, color='black', **kwargs):
    "Set title of `ax` to `o`, or print `o` if `ax` is `None`"
    ax = ifnone(ax,ctx)
    if ax is None: print(o)
    elif hasattr(ax, 'set_title'):
        t = ax.title.get_text()
        if len(t) > 0: o = t+'\n'+str(o)
        ax.set_title(o, color=color)
    elif isinstance(ax, pd.Series):
        while title in ax: title += '_'
        ax = ax.append(pd.Series({title: o}))
    return ax

class MyShowTitle:
    "Base class that adds a simple `show`"
    
    @classmethod
    def init(cls, data, **kwargs):
        item = cls(data)
        item._show_args = kwargs
        return item

    def show(self, ctx=None, **kwargs):
        "Show self"
        return my_show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))

# it seems that python prioritising prior inherited class when finding methods   

class MyTitledInt(MyShowTitle, Int): pass

class MyTitledFloat(MyShowTitle, Float): pass

# I created it
class MyTitledBool(MyShowTitle, Int): # python says bool can't be base class
    def show(self, ctx=None, **kwargs):
        "Show self"
        return my_show_title(str(bool(self)), ctx=ctx, **merge(self._show_args, kwargs))

class MyTitledStr(MyShowTitle, Str):
  def truncate(self, n):
    "Truncate self to `n`"
    words = self.split(' ')[:n]
    return MyTitledStr.init(' '.join(words), **self._show_args)

class MyTitledTuple(MyShowTitle, Tuple): pass

class MyCategory(MyShowTitle, Str): pass

class MyMultiCategory(MyShowTitle, L):
    def show(self, ctx=None, sep=';', color='black', **kwargs):
        return my_show_title(sep.join(self.map(str)), ctx=ctx, color=color, **merge(self._show_args, kwargs))

""" Caution !!
These two function is inperfect.
But they cope with mutiple input columns problem (n_inp >1), which cause no df printing but just sequentail print
These will be a problem when you are doing non-text problem with n_inp > 1 (multiple input column),
which shouldn't be the case of huggingface/nlp user.
And I hope fastai come up with a good solution to show_batch multiple inputs problems for text/non-text.
"""
@typedispatch
def show_batch(x:tuple, y, samples, ctxs=None, max_n=9, **kwargs):
  if ctxs is None: ctxs = get_empty_df(min(len(samples), max_n))
  ctxs = show_batch[object](x, y, samples, max_n=max_n, ctxs=ctxs, **kwargs)
  display_df(pd.DataFrame(ctxs))
  return ctxs

@typedispatch
def show_results(x: tuple, y, samples, outs, ctxs=None, max_n=10, trunc_at=150, **kwargs):
  if ctxs is None: ctxs = get_empty_df(min(len(samples), max_n))
  ctxs = show_results[object](x, y, samples, outs, ctxs=ctxs, max_n=max_n, **kwargs)
  display_df(pd.DataFrame(ctxs))
  return ctxs

class HF_Dataset():
  "A wrapper for `nlp.Dataset` to make it fulfill behaviors of `fastai2.Datasets` we need. Unless overidded, it behaves like original `nlp.Dataset`"
  
  def __init__(self, hf_dset, cols=None, hf_toker=None, neat_show=False, n_inp=1):
    """
    Args:
      `hf_dset` (`nlp.Dataset`)
      `cols` (`Optional`, default: `None`): **specify columns whose values form a output sample in order**, and the semantic type of each column to encode/decode, with one of the following signature.
      - `cols`(`Dict[Fastai Semantic Tensor]`): encode/decode {key} columns with {value} semantic tensor type. If {value} is `noop`, regard it as `TensorTuple` by default.
      - `cols`(`List[str]`): 
        - if of length 1, regard the 1st element as `TensorText`
        - if of length 2, regard the 1st element as `TensorText`, 2nd element as `TensorCategory`
        - Otherwise, regard all elements as `TensorTuple`
      - `None`: use `hf_dset.column_names` and deal with it like `List[str]` above.
      `hf_toker`: tokenizer of HuggingFace/Transformers
      `neat_show` (`Optional[bool]`, default:`False`): Show the original sentence instead of tokens joined.
      `n_inp (`int`, default:1) the first `n_inp` columns of `cols` is x, and the rest is y .
    """
    
    # some default setting for tensor type used in decoding
    if cols is None: cols = hf_dset.column_names
    if isinstance(cols, list): 
      if n_inp==1: 
        if len(cols)==1: cols = {cols[0]: TensorText}
        elif len(cols)==2: cols = {cols[0]: TensorText, cols[1]: TensorCategory}
      else: cols = { c: noop for c in cols }
    assert isinstance(cols, dict)
    
    # make dataset output pytorch tensor
    if hf_dset.format['type'] != 'torch': 
      hf_dset.set_format( type='torch', columns=list(cols.keys()) )

    # store attributes
    self.pad_idx = hf_toker.pad_token_id
    store_attr(self, "hf_dset,cols,n_inp,hf_toker,neat_show")

  def __getitem__(self, idx):
    sample = self.hf_dset[idx]
    return tuple( tensor_cls(sample[col]) for col, tensor_cls in self.cols.items() )

  def __len__(self): return len(self.hf_dset)

  @property
  def col_names(self): return list(self.cols.keys())

  def decode(self, o, full=True): # `full` is for micmic `Dataset.decode` 
    if len(self.col_names) != len(o): return tuple( self._decode(o_) for o_ in o )
    return tuple( self._decode(o_, self.col_names[i]) for i, o_ in enumerate(o) )

  def _decode_title(self, d, title_cls, title): 
    if title: return title_cls.init(d, title=title)
    else: return title_cls.init(d)

  @typedispatch
  def _decode(self, t:torch.Tensor, title):
    if t.shape: title_cls = MyTitledTuple
    elif isinstance(t.item(),bool): title_cls = MyTitledBool # bool is also int, so check whether is bool first
    elif isinstance(t.item(),float): title_cls = MyTitledFloat
    elif isinstance(t.item(),int): title_cls = MyTitledInt
    return self._decode_title(t.tolist(), title_cls , title)

  @typedispatch
  def _decode(self, t:TensorText, title): 
    assert self.hf_toker, "You should give a huggingface tokenizer if you want to show batch."
    if self.neat_show: text = self.hf_toker.decode([idx for idx in t if idx != self.hf_toker.pad_token_id])
    else: text = ' '.join(self.hf_toker.convert_ids_to_tokens(t))
    return self._decode_title(text, MyTitledStr, title)

  @typedispatch
  def _decode(self, t:LMTensorText, title): return self._decode[TensorText](self, t, title)

  @typedispatch
  def _decode(self, t:TensorCategory, title): return self._decode_title(t.item(), MyCategory, title)

  @typedispatch
  def _decode(self, t:TensorMultiCategory, title): return self._decode_title(t.tolist(), MyMultiCategory, title)

  def __getattr__(self, name):
    "If not defined, let the nlp.Dataset in it act for us."
    if name in HF_Dataset.__dict__: return HF_Dataset.__dict__[name]
    elif name in self.__dict__: return self.__dict__[name]
    elif hasattr(self.hf_dset, name): return getattr(self.hf_dset, name)
    raise AttributeError(f"Both 'HF_Dataset' object and 'nlp.Dataset' object have no '{name}' attribute ")
  
class HF_Datasets(FilteredBase):
  _dl_type,_dbunch_type = MySortedDL,DataLoaders
  
  @delegates(HF_Dataset.__init__)
  def __init__(self, hf_dsets: dict, test_with_label=False, **kwargs):
    """
    Args:
      `hf_dsets` (`Dict[nlp.Dataset]`): the order of dict items will be the order of `HF_Dataloader`s
      `test_with_label` (`bool`, default:`False`): whether testset come with labels.
    """
    cols, n_inp = kwargs.pop('cols', None), kwargs.get('n_inp', 1)
    self.hf_dsets = {};
    for split, dset in hf_dsets.items():
      if cols is None: cols = dset.column_names
      if split.startswith('test') and not test_with_label: 
        if isinstance(cols, list): _cols = cols[:n_inp]
        else: _cols = { k:v for _, (k,v) in zip(range(n_inp),cols.items()) }
      else: _cols = cols
      self.hf_dsets[split] = HF_Dataset(dset, cols=_cols, **kwargs)

  def subset(self, i): return list(self.hf_dsets.values())[i]
  def __getitem__(self, split): return self.hf_dsets[split]
  @property
  def n_subsets(self): return len(self.hf_dsets)
  @property
  def cache_dir(self): return Path(next(iter(self.hf_dsets.values())).cache_files[0]['filename']).parent
  
  @delegates(FilteredBase.dataloaders)
  def dataloaders(self, device='cpu', cache_dir=None, cache_name=None, **kwargs):
    """
    Args:
      device (str, default:`'cpu'`)
      cache_dir (`Optional[str]`, default: `None`): directory to store dataloader caches. if `None`, use cache directory of first `nlp.Dataset` stored.
      cache_name (`Optional[str]`, default: `None`): format string with only one param `{split}` as cache file name under `cache_dir` for each split. If `None`, use autmatical cache path by hf/nlp.      
    """
    dl_kwargs = kwargs.pop('dl_kwargs', [{} for _ in range(len(self.hf_dsets))])
    # infer cache file names for each dataloader if needed
    dl_type = kwargs.pop('dl_type', self._dl_type)
    if dl_type==MySortedDL and cache_name:
      assert "{split}" in cache_name, "`cache_name` should be a string with '{split}' in it to be formatted."
      cache_dir = Path(cache_dir) if cache_dir else self.cache_dir
      cache_dir.mkdir(exist_ok=True)
      if not cache_name.endswith('.json'): cache_name += '.json'
      for i, split in enumerate(self.hf_dsets):
        dl_kwargs[i]['cache_file'] = cache_dir/cache_name.format(split=split)
    # change default to not drop last
    kwargs['drop_last'] = kwargs.pop('drop_last', False)
    # when corpus like glue/ax has only testset, set it to non-train setting
    if list(self.hf_dsets.keys())[0].startswith('test'):
      kwargs['shuffle_train'] = False
      kwargs['drop_last'] = False
    return super().dataloaders(dl_kwargs=dl_kwargs, device=device, **kwargs)

class HF_MergedDataset():
  def __init__(self, *datasets):
    self.dsets = datasets
    self.len = reduce(lambda a,d: a+len(d), self.dsets, 0)
  def __len__(self):
    return self.len
  def __getitem__(self, i):
    for dset in self.dsets:
      if i < len(dset): return dset[i]
      else: i -= len(dset)
    raise IndexError
  def set_format(self, type, columns):
    for dset in self.dsets: dset.set_format(type, columns)
  @property
  def format(self):
    form = self.dsets[0].format
    for dset in self.dsets:
      assert form == dset.format
    return form
  @property
  def cache_files(self):
    return concat(*[ds.cache_files for ds in self.dsets])