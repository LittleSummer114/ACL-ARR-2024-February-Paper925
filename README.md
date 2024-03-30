**Case 1**

the **constituent tree** of "Oppo 's flagship machine has good quality control and texture .":

```py
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
```

the **semantic dependency graph** of "Oppo 's flagship machine has good quality control and texture .":

```py

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
```


**Case 2**

the **constituent tree** of the dialogue "But it 's useless, software optimization sucks", "Optimization is Xiaomi 's weakness [ allow sadness][allow sadness][allow sadness ]"

```py
                                                                   TOP                                                                                
                                                                    |                                                                                  
                                                                    S                                                                                 
  __________________________________________________________________|_______________________________________________________________________________   
 |   |   |     |               |                 |                       S                      |              S                                    | 
 |   |   |     |               |                 |          _____________|_____                 |              |                                    |  
 |   |   |     |               |                 |         |                   VP               |              VP                                   | 
 |   |   |     |               |                 |         |         __________|___             |     _________|________________________            |  
 |   |   |     |               |                 |         |        |              NP           |    |         |                        VP          | 
 |   |   |     |               |                 |         |        |           ___|_____       |    |         |               _________|_____      |  
 |   NP  |    ADJP             NP                |         NP       |          NP        |      |    |         |              |               NP    | 
 |   |   |     |         ______|_______          |         |        |     _____|___      |      |    |         |              |               |     |  
 _   _   _     _        _              _         _         _        _    _         _     _      _    _         _              _               _     _ 
 |   |   |     |        |              |         |         |        |    |         |     |      |    |         |              |               |     |  
But  it  's useless, software     optimization sucks, Optimization  is Xiaomi      's weakness  [  allow sadness][allow sadness][allow     sadness  ] 

```
the **semantic dependency graph** of "The workmanship and the screen are indeed good, but sometimes it will suddenly get stuck":

```py

1	But	_	_	_	_	_	_	0:root	_
2	it	_	_	_	_	_	_	_	_
3	's	_	_	_	_	_	_	_	_
4	useless,	_	_	_	_	_	_	_	_
5	software	_	_	_	_	_	_	_	_
6	optimization	_	_	_	_	_	_	6:compound	_
7	sucks,	_	_	_	_	_	_	_	_
8	Optimization	_	_	_	_	_	_	_	_
9	is	_	_	_	_	_	_	_	_
10	Xiaomi	_	_	_	_	_	_	_	_
11	's	_	_	_	_	_	_	_	_
12	weakness	_	_	_	_	_	_	12:ARG1	_
13	[	_	_	_	_	_	_	_	_
14	allow	_	_	_	_	_	_	_	_
15	sadness][allow	_	_	_	_	_	_	_	_
16	sadness][allow	_	_	_	_	_	_	_	_
17	sadness	_	_	_	_	_	_	_	_
18	]	_	_	_	_	_	_	_	_
```

```py

               TOP                                                                                     
                |                                                                                       
                S                                                                                      
       _________|_________________                                                                      
      |                           VP                                                                   
      |                 __________|________                                                             
      |                |                   PP                                                          
      |                |      _____________|____________________                                        
      |                |     |                                  NP                                     
      |                |     |     _____________________________|__________                             
      |                |     |    |   |    |       |                      SBAR                         
      |                |     |    |   |    |       |        _______________|______                      
      |                |     |    |   |    |       |       |                      S                    
      |                |     |    |   |    |       |       |                      |                     
      |                |     |    |   |    |       |       |                      VP                   
      |                |     |    |   |    |       |       |     _________________|_______              
      |                |     |    |   |    |       |       |    |   |      |              PP           
      |                |     |    |   |    |       |       |    |   |      |       _______|___          
      NP               |     |    |   |    |       |      WHNP  |  ADVP   ADJP    |           NP       
  ____|_________       |     |    |   |    |       |       |    |   |      |      |    _______|_____    
 _    _    _    _      _     _    _   _    _       _       _    _   _      _      _   _       _     _  
 |    |    |    |      |     |    |   |    |       |       |    |   |      |      |   |       |     |   
The redmi k20 series starts with the  K  series borders, which are also narrower  at the     same price



1	The	_	_	_	_	_	_	_	_
2	redmi	_	_	_	_	_	_	_	_
3	k20	_	_	_	_	_	_	3:ARG1	_
4	series	_	_	_	_	_	_	1:BV|4:ARG1	_
5	starts	_	_	_	_	_	_	6:ARG1	_
6	with	_	_	_	_	_	_	_	_
7	the	_	_	_	_	_	_	_	_
8	K	_	_	_	_	_	_	_	_
9	series	_	_	_	_	_	_	_	_
10	borders,	_	_	_	_	_	_	9:compound	_
11	which	_	_	_	_	_	_	_	_
12	are	_	_	_	_	_	_	_	_
13	also	_	_	_	_	_	_	_	_
14	narrower	_	_	_	_	_	_	13:ARG1|15:ARG1	_
15	at	_	_	_	_	_	_	_	_
16	the	_	_	_	_	_	_	_	_
17	same	_	_	_	_	_	_	_	_
18	price	_	_	_	_	_	_	_	_
```
