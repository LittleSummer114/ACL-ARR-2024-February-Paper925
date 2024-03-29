
We show the process of obtaining the constituent tree and semantic dependency graph of a given dialogue here.

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

>>> sdp = dep_parser = Parser.load('sdp-vi-en')
>>> sdp.predict([[('I','I','PRP'), ('saw','see','VBD'), ('Sarah','Sarah','NNP'), ('with','with','IN'),
                  ('a','a','DT'), ('telescope','telescope','NN'), ('.','_','.')]],
                verbose=False)[0]

1	Oppo	_	_	_	_	_	_	_	_
2	's	_	_	_	_	_	_	_	_
3	flagship	_	_	_	_	_	_	_	_
4	machine	_	_	_	_	_	_	2:compound|3:compound|5:ARG1	_
5	has	_	_	_	_	_	_	0:root	_
6	good	_	_	_	_	_	_	_	_
7	quality	_	_	_	_	_	_	_	_
8	control	_	_	_	_	_	_	5:ARG2|6:ARG1|7:compound	_
9	and	_	_	_	_	_	_	_	_
10	texture	_	_	_	_	_	_	8:_and_c	_
11	.	_	_	_	_	_	_	_	_





                                                  TOP                                               
                                                    |                                                 
                                                    S                                                
                          __________________________|______________________                           
                         |                               |                 S                         
                         |                               |       __________|______                    
                         S                               |      |      |          VP                 
                      ___|___________________            |      |      |    ______|__________         
                     NP                      VP          |      |      |   |      |          VP      
      _______________|_______            ____|______     |      |      |   |      |       ___|____    
     NP              |       NP         |   ADVP   ADJP  |     ADVP    NP  |     ADVP    |       ADJP
  ___|_______        |    ___|____      |    |      |    |      |      |   |      |      |        |   
 _           _       _   _        _     _    _      _    _      _      _   _      _      _        _  
 |           |       |   |        |     |    |      |    |      |      |   |      |      |        |   
The     workmanship and the     screen are indeed good, but sometimes  it will suddenly get     stuck


