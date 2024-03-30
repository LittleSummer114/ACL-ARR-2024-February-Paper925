**Case 1**

the **constituent tree** of "But it 's useless, software optimization sucks", "Optimization is Xiaomi 's weakness [ allow sadness][allow sadness][allow sadness ]"

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
the **semantic dependency graph** of "But it 's useless, software optimization sucks", "Optimization is Xiaomi 's weakness [ allow sadness][allow sadness][allow sadness ]":

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

**Case 2**

the **constituent tree** of the dialogue "Daily photography is still inferior to iPhone and Samsung"

```py
                                  TOP                              
                                   |                                
                                   S                               
        ___________________________|______                          
       |                                  VP                       
       |                __________________|____                     
       |               |    |                 ADJP                 
       |               |    |       ___________|_____               
       |               |    |      |                 PP            
       |               |    |      |       __________|___           
       NP              |   ADVP    |      |              NP        
   ____|_______        |    |      |      |     _________|_____     
  _            _       _    _      _      _    _         _     _   
  |            |       |    |      |      |    |         |     |    
Daily     photography  is still inferior  to iPhone     and Samsung

```
the **semantic dependency graph** of "Daily photography is still inferior to iPhone and Samsung":

```py

1	Daily	_	_	_	_	_	_	_	_
2	photography	_	_	_	_	_	_	1:ARG1|5:ARG1	_
3	is	_	_	_	_	_	_	_	_
4	still	_	_	_	_	_	_	_	_
5	inferior	_	_	_	_	_	_	0:root|4:ARG1|6:ARG1	_
6	to	_	_	_	_	_	_	_	_
7	iPhone	_	_	_	_	_	_	6:ARG2	_
8	and	_	_	_	_	_	_	_	_
9	Samsung	_	_	_	_	_	_	_	_
```
