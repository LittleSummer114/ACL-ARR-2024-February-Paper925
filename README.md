
We show four cases to explain why **constituent tree** and **semantic dependency graph** should be incorporated **dynamically**:

| Case | constituent tree | semantic dependency graph |

| 1 | **×** | **√** |

| 2 | **√** | **×** |

| 3 | **√** | **×** |

| 4 | **√** | **×** |
 
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
2	## 's	_	_	_	_	_	_	_	_
3	## flagship	_	_	_	_	_	_	_	_
4	## machine	_	_	_	_	_	_	2:compound|3:compound|5:ARG1	_
5	has	_	_	_	_	_	_	0:root	_
6	good	_	_	_	_	_	_	_	_
7	quality	_	_	_	_	_	_	_	_
8	control	_	_	_	_	_	_	5:ARG2|6:ARG1|7:compound	_
9	and	_	_	_	_	_	_	_	_
10	texture	_	_	_	_	_	_	8:_and_c	_
11	.	_	_	_	_	_	_	_	_
```

**Case 2**

the **constituent tree** of "The workmanship and the screen are indeed good, but sometimes it will suddenly get stuck":

```py
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
The     workmanship and the     screen are **indeed good**, but sometimes  it will suddenly get     stuck

```

the **semantic dependency graph** of "The workmanship and the screen are indeed good, but sometimes it will suddenly get stuck":

```py

1	The	_	_	_	_	_	_	_	_
2	workmanship	_	_	_	_	_	_	1:BV|8:ARG1	_
3	and	_	_	_	_	_	_	_	_
4	the	_	_	_	_	_	_	_	_
5	screen	_	_	_	_	_	_	2:_and_c|4:BV	_
6	are	_	_	_	_	_	_	_	_
7	**indeed**	_	_	_	_	_	_	_	_
8	**good**	_	_	_	_	_	_	7:ARG1	_
9	,	_	_	_	_	_	_	_	_
10	but	_	_	_	_	_	_	_	_
11	sometimes	_	_	_	_	_	_	_	_
12	it	_	_	_	_	_	_	15:ARG1	_
13	will	_	_	_	_	_	_	_	_
14	suddenly	_	_	_	_	_	_	_	_
15	get	_	_	_	_	_	_	11:ARG1|14:ARG1	_
16	stuck	_	_	_	_	_	_	_	_

```
