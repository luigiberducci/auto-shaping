import os
import pathlib
from lark import Lark

from shaping.parser.transformer import RewardShapingTransformer

# load grammar
wd = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
with open(wd / 'grammar.txt', 'r') as f:
    grammar = f.read()

# create parser
my_parser = Lark(grammar, start='start', parser='lalr', transformer=RewardShapingTransformer())

# example
texts = [
    '["x_limit"=0.5, ensure "x">"x_limit", achieve "x"<=0.5, encourage "y">5.0]',
    'achieve "xy" == 9.99',
    '[achieve "xy" == 9.99, ensure "x" > 0.5]',
    ]

for text in texts:
    result = my_parser.parse(text)
    print(result.pretty())