import pylearn2
from pylearn2.config import yaml_parse
import argparse

if __name__ == '__main__' :
  parser = argparse.ArgumentParser()
  parser.add_argument("yaml_filepath", help = "yaml file path")
  args = parser.parse_args()
  yaml_filepath = args.yaml_filepath
  model = yaml_parse.load_path(yaml_filepath)
  model.main_loop()