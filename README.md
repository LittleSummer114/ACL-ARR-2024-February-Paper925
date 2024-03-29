
Case Study.

```py
>>> from supar import Parser
# if the gpu device is available
# >>> torch.cuda.set_device('cuda:0')  
>>> parser = Parser.load('dep-biaffine-en')
>>> dataset = parser.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=False)
>>> con = Parser.load('con-crf-en')
>>> con.predict(['I', 'saw', 'Sarah', 'with', 'a', 'telescope', '.'], verbose=False)[0].pretty_print()
              TOP                       
               |                         
               S                        
  _____________|______________________   
 |             VP                     | 
 |    _________|____                  |  
 |   |    |         PP                | 
 |   |    |     ____|___              |  
 NP  |    NP   |        NP            | 
 |   |    |    |     ___|______       |  
 _   _    _    _    _          _      _ 
 |   |    |    |    |          |      |  
 I  saw Sarah with  a      telescope  . 

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
