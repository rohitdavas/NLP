

# Machine Translation

## Overview and the general formulation

![translation overview](F:\Github\NLP\Notes\Images\MT1.png)

Say, you want to translate the English word to French word. 

One way to do it :

1. find word embedding of the word in English

2. find the corresponding word's french word embeddings

3. Transform somehow, by a rule that generalizes for any English word later. 

   using $$ X * R = Y $$ we find R such that it satisfies the relation. 

   

![image-20200907114424654](Machine%20Translation.assets/image-20200907114424654.png)

- Solve for R 
  - initialise R
  - in a loop:
    - loss = || XR - Y ^||2^ ~F~  # frobenious norm square
    - $$g = \frac {d loss}{dR}$$ 
    - $$R = R - alpha * g $$
    - break if loss falls below a threshold. 

**Frobenius norm **: ![image-20200907115733539](Machine%20Translation.assets/image-20200907115733539.png)

**gradient of Frobenius norm sqaure **![image-20200907115646478](Machine%20Translation.assets/image-20200907115646478.png)

​		