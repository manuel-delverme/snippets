def to_tuples(list_of_lists):
  tuple_of_tuples = []
  for item in list_of_lists:
    if isinstance(item, list):
      item = to_tuples(item)
    tuple_of_tuples.append(item)
  return tuple(tuple_of_tuples)


def disk_cache(f: Callable) -> _lru_cache_wrapper:
  @lru_cache(maxsize=1024)
  def wrapper(*args, **kwargs):
    fid = f.__name__
    cache_file = "cache/{}".format(fid)
    if args:
      if not os.path.exists(cache_file):
        os.makedirs(cache_file)
      fid = fid + "/" + "::".join(str(arg) for arg in args).replace("/", "_")
      cache_file = "cache/{}".format(fid)
    cache_file += ".pkl"
    try:
      with open(cache_file, "rb") as fin:
        retr = pickle.load(fin)
    except FileNotFoundError:
      retr = f(*args, **kwargs)
      with open(cache_file, "wb") as fout:
        pickle.dump(retr, fout)
    return retr

  return wrapper


@contextmanager
def suppress_stdout():
  with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
      yield
    finally:
      sys.stdout = old_stdout


def one_hot_actions(actions: Union[Tuple[Tuple[int, int]], Tuple[Tuple[int, int, int]]], num_actions: int) -> ndarray:
  possible_actions = np.zeros((len(actions), num_actions), np.float32)
  for row, pas in enumerate(actions):
    for pa in pas:
      possible_actions[row, pa] = 1
  return possible_actions

