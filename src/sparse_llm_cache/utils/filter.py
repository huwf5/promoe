import re
class Filter:
  def __init__(self):
    pass
  def __call__(self, *args, **kwds):
    return True
  def reverse(self):
    return ReverseFilter(self)

class ReverseFilter(Filter):
  def __init__(self, filter : Filter):
    self.filter = filter
  def __call__(self, *args, **kwds):
    return not self.filter(*args, **kwds)

class RegexFilter(Filter):
  def __init__(self, pattern):
    self.pattern = pattern
  def __call__(self, name, *args, **kwds):
    return re.match(self.pattern, name) != None
