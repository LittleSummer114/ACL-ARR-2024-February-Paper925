
We show the process of obtaining the constituent tree of each dialogue here.

```py

from supar import Parser





>>> from supar import Parser
>>> con_parser = Parser.load('con-crf-roberta-en')
>>> token = "Oppo 's flagship machine has good quality control and texture .".split(' ')
>>> con_parser.predict(token, verbose=False)[0].pretty_print()

                                  TOP                                 
                                   |                                   
                                   S                                  
           ________________________|________________________________   
          NP                              VP                        | 
       ___|______________       __________|_______                  |  
      NP        |        |     |                  NP                | 
  ____|___      |        |     |    ______________|___________      |  
 _        _     _        _     _   _      _       _     _     _     _ 
 |        |     |        |     |   |      |       |     |     |     |  
Oppo      's flagship machine has good quality control and texture  . 

>>> sdp = Parser.load('sdp-biaffine-en')
>>> sdp.predict([[('I','I','PRP'), ('saw','see','VBD'), ('Sarah','Sarah','NNP'), ('with','with','IN'),
                  ('a','a','DT'), ('telescope','telescope','NN'), ('.','_','.')]],
                verbose=False)[0]
1       I       I       PRP     _       _       _       _       2:ARG1  _
2       saw     see     VBD     _       _       _       _       0:root|4:ARG1   _
3       Sarah   Sarah   NNP     _       _       _       _       2:ARG2  _
4       with    with    IN      _       _       _       _       _       _
5       a       a       DT      _       _       _       _       _       _
6       telescope       telescope       NN      _       _       _       _       4:ARG2|5:BV     _
7       .       _       .       _       _       _       _       _       _
