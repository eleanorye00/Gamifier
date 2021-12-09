import re

def to_snake_case(camel_case):
  """Returns a CamelCase string as a snake_case string."""
  if not re.match(r"^[A-Za-z_]\w*$", camel_case):
    raise ValueError(
        "Input string %s is not a valid Python identifier." % camel_case)
  # Add underscore at word start and ends.
  underscored = re.sub(r"([A-Z][a-z])", r"_\1", camel_case)
  underscored = re.sub(r"([a-z])([A-Z])", r"\1_\2", underscored)
  # Add underscore before alphanumeric chunks.
  underscored = re.sub(r"([a-z])([0-9][^_]*)", r"\1_\2", underscored)
  # Remove any underscores at start or end of name and convert to lowercase.

  return underscored.strip("_").lower()


