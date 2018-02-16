import heapq

# from tf/models/im2txt
class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
          heapq.heappush(self._data, x)
        else:
          heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
          data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []

# from tf/models/im2txt
class Beam(object):
    """Used for beam_search"""
    def __init__(self, sentence, state, logprob, score):
        self.sentence = sentence
        self.logprob = logprob
        self.state = state
        self.score = score

    def __cmp__(self, other):
        """Compares captions by score."""
        assert isinstance(other, Beam)
        if self.score == other.score:
          return 0
        elif self.score < other.score:
          return -1
        else:
          return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Beam)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Beam)
        return self.score == other.score
