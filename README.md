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
