---
layout: post
title: Bayes theorem (solving Monty Hall problem)
date: 2017-12-09 13:32
comments: true
external-url:
categories: bayes
---

> Monty Hall problem is a famous probability puzzle: You are in a game show, and in front of you are three doors: Behind one of them there is a car and behind the others goats. You pick a door (1 or 2 or 3) and the hosts, who knows what's behind doors, opens another door, that has a goat behind. Then asks you: "Do you keep to choose the same door?", it is your advantage to switch your door?

![monty hall show](/assets/monty.png)

This could be one of the intuitive ways to determinte every case and compute the sum of each result:

**door** : initial door picked  
**car** : door where the car is hiden  
**switch** : action to switch 0=Stick 1=Change  
**result** : result of the game 0=goat 1=car  

| door     | car | switch | result |
|:--------:|:---:|:------:|:------:|
| 1        | 1   | 0      | 1      |
| 1        | 1   | 1      | 0      |
| 1        | 2   | 0      | 0      |
| 1        | 2   | 1      | 1      |
| 1        | 3   | 0      | 0      |
| 1        | 3   | 1      | 1      |
| 2        | 2   | 0      | 1      |
| 2        | 2   | 1      | 0      |
| 2        | 3   | 0      | 0      |
| 2        | 3   | 1      | 1      |
| 2        | 1   | 0      | 0      |
| 2        | 1   | 1      | 1      |
| 3        | 3   | 0      | 1      |
| 3        | 3   | 1      | 0      |
| 3        | 1   | 0      | 0      |
| 3        | 1   | 1      | 1      |
| 3        | 2   | 0      | 0      |
| 3        | 2   | 1      | 1      |

 $$ \text{total cases}  = {18}.$$  
 $$ \text{total cases with car result}  = {9}.$$  
 $$ \text{total cases that stick with car result}  = {3}.$$  
 $$ \text{total cases that change with car result}  = {6}.$$  
 $$ \text{car probability that stick}  = {3 \over 9}.$$  
 $$ \text{car probability that change}  = {6 \over 9}.$$  

 There is a formal view to calculate the result with bayes theorem:

$$P(A|B)  = {P(B|A)*P(A) \over P(B)}.$$

There is a more accurate form if you consider that the sum:

$$\sum_{i=1}^N{P(A_i)}=1$$

$$P(A_k|B)  = {P(B|A_k)*P(A_k) \over \sum_{i=1}^N{P(B|A_i)*P(A_i)}}.$$

In our case let's assume we choose door #1  
**C** = where the car really is hidden

A priori probabilities:

$$P(C=1) = P(C=2) = P(C=3) = {1 \over 3}$$

**H** = door that Monty Hall opens  
Assume he opens door #2 without loss of generality, problem is symmetric

First case Monty has half probability to choose door 2 or 3, both have a goat behind:  

$$P(H=2|C=1)= {1 \over 2}$$

Second case Monty has to choose the door #3, because door #2 has de car:  

$$P(H=2|C=2)= 0$$

Third case Monty has to choose the door #2, because door #3 has the car:  

$$P(H=2|C=3)= 1$$

Now we want to know whether to stick with door #1 or switch to door #3 (Monty has opened door #2)

$$\text{Probability if stick in #1 door}=P(C=1|H=2)$$

$$P(C=1|H=2)={P(H=2|C=1)*P(C=1) \over{P(H=2|C=1)*P(C=1)+P(H=2|C=2)*P(C=2)+P(H=2|C=3)*P(C=3)}}.$$

$$P(C=1|H=2)={(1/2)*(1/3) \over {(1/2) * (1/3) + 0 * (1/3)  + 1 * (1/3)}}= {1 \over 3}$$

$$\text{Probability if change to #3 door}=P(C=3|H=2)$$

$$P(C=3|H=2)={P(H=2|C=3)*P(C=3) \over{P(H=2|C=3)*P(C=1)+P(H=2|C=2)*P(C=2)+P(H=2|C=3)*P(C=3)}}.$$

$$P(C=3|H=2)={1*(1/3) \over {(1/2) * (1/3) + 0 * (1/3)  + 1 * (1/3)}}= {2 \over 3}$$






