"""
Utility functions.
"""

from typing import Union
from collections import defaultdict


class ShortPrefix:
    """Map unique prefixes to words.

    Usage:
    >>> sp = ShortPrefix(["apple", "banana", "apricot", "berry"])
    >>> sp.match("app")  # Returns "apple"
    >>> sp.match("ap")  # Returns [] (ambiguous)
    >>> sp.match("ban")  # Returns "banana"
    >>> sp.match("b")  # Returns None (ambiguous)
    >>> sp.could_be("ap")  # Returns ["apple", "apricot"]
    >>> sp.could_be("b")  # Returns ["banana", "berry"]
    >>> sp.could_be("c")  # Returns [] (no matches)
    """

    def __init__(self, words, lowercase: bool = True):
        """Construct with a list of words.

        Args:
            words (list of str): List of words to generate unique prefix mappings for.
            lowercase (bool): If True, convert all words to lowercase before processing.
        """
        if lowercase:
            wd = [w.lower() for w in words]
            self._wd_map = {w.lower(): w for w in words}
        else:
            wd = words
        self._words = words
        pos = 1
        self._map = {}
        while wd:
            # build a map of prefixes (of len 'pos') to words
            pfx_map = defaultdict(list)
            for w in wd:
                pfx_map[w[:pos]].append(w)
            # add unique prefix/word pairs to the map,
            # and put the rest back into the list for the next iteration
            wd = []
            for k, v in pfx_map.items():
                if len(v) == 1:
                    self._map[k] = v[0]
                else:
                    wd.extend(v)
            pos += 1
        self._lower = lowercase

    @property
    def words(self) -> list[str]:
        """Get the list of words that were used to construct the prefix map."""
        return self._words

    def match(self, s: str) -> Union[str, None]:
        """Find the unique word for a given prefix string.

        Returns:
            str or None: The unique word that matches the prefix, or None if no unique word, or any word, matches.
        """
        if self._lower:
            s = s.lower()
        n = len(s)
        for k, v in self._map.items():
            m = len(k)
            if n >= m and s.startswith(k) and v.startswith(s):
                if self._lower:
                    return self._wd_map[v]
                return v
        return None

    def could_be(self, s: str) -> list[str]:
        """Find all words that _could_ match a given prefix string.

        Note that this will not return words that are unambiguous matches.
        Put another way, if :meth:`match` returns a word for this input,
        then this method will return an empty list.

        Returns:
            list of str: A list of words that could match the prefix.
        """
        if self._lower:
            s = s.lower()
        result = []
        n = len(s)
        for k, v in self._map.items():
            m = len(k)
            if n < m and k.startswith(s):
                if self._lower:
                    result.append(self._wd_map[v])
                else:
                    result.append(v)
        return result
