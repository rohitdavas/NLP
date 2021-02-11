---
typora-copy-images-to: Images
---



# PCA and the Visualization

 References : [Coursera NLP Specialization Course 1](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/ungradedLab/n1iZS/another-explanation-about-pca) 

<u>PCA is credited to Karl Pearson, 1901.</u> 

It is often the case that you'll end up having vectors in very, very high dimensions.  You want to find a way to reduce the dimension of these vectors to two dimensions so you can plot it on an XY axis. 

![PCA 1 ](F:\Github\NLP\Notes\Images\PCA 1 -1599362055735.png)

Imagine you have the above representation for your words. clearly, there is a relationship b/w oil and gas, and city and town. And you want to visualize that this relationship is being captured by the representation. 

Now, dimensionality reduction is one choice. ![PCA 2](F:\Github\NLP\Notes\Images\PCA 2.png)

Now you can plot a visual of your words. 

![PCA 3](F:\Github\NLP\Notes\Images\PCA 3.png)

------

## How it works ? 

For the sake of simplicity, I'll begin with a two dimensional vector space.  Say that you want your data to be represented by one feature instead.  Using PCA, first you'll find a set of uncorrelated features. 

And then projects your data to a one dimensional space, trying to retain as much information as possible

![image-20200906085219641](F:\Github\NLP\Notes\Images\image-20200906085219641.png)

------



## PCA Algorithm

1. Mean normalize your data. ![image-20200906085957707](F:\Github\NLP\Notes\Images\image-20200906085957707.png)

2. Get covariance matrix . ![image-20200906090042027](F:\Github\NLP\Notes\Images\image-20200906090042027.png)

3. Perform SVD. ![image-20200906090104484](F:\Github\NLP\Notes\Images\image-20200906090104484.png).

    It gives us three matrices

   ![image-20200906090147764](F:\Github\NLP\Notes\Images\image-20200906090147764.png)

4. Dot product X with Eigenvectors by selecting no. of dimensions we want to reduce it to. 

   For visualization it's better to use two dimension.

   ![image-20200906090433876](F:\Github\NLP\Notes\Images\image-20200906090433876.png)

5. You can analyze the percentage of variance retained by above formula. 

   

**NOTE :** *Your eigenvectors and eigenvalues should be organized according to the eigenvalues in descending order.* But most of the time libraries do if for you. 



------

## Applications

I tried to collect the various applications of PCA, but it's not an exhaustive list.

1. simple space transformation
2. dimensionality reduction
3. mixture separation from spectral information 



------

## Code 1 : Understanding the PCA 

generate data with dependency as y = n*x , where x belongs to the uniform random distribution and than transform use PCA. 

As we see,  PCA will remove the redundancy of features. 

```python
from sklearn.decomposition import PCA      # PCA library
import numpy as np 

#----------------------------
# generate some unifrom data
#-----------------------------
n = 1  # The amount of the correlation
x = np.random.uniform( low = 1, high = 2, size = 1000) 
y = x.copy() * n # Make y = n * x

#---------------------------------------
# PCA works better if data is centered
#----------------------------------------
x = x - np.mean( x )
y = y - np.mean ( y )


#-----------------------
# create a data frame 
#------------------------
data = pd.Dataframe({'x':x, 'y':y})
plt.scatter(data.x, data.y)
```

![image](F:\Github\NLP\Notes\Images\PCA data.png)



```python
#------------------------------
# initialise the PCA algorithm
#--------------------------------
pca = PCA ( n_components = 2 )
pcaTr = pca.fit ( data )

#-------------------------------
# transform the data base
# on the rotation matrix of pcaTr
#--------------------------------
rotatedData = pcaTr.transform(data)

# new data
dataPCA = pd.DataFrame(data = rotatedData, columns = ['PC1', 'PC2'])

# plot the dataPCA
plt.scatter(dataPCA.PC1, dataPCA.PC2)
plt.show()
```

![PCA rotated output ](F:\Github\NLP\Notes\Images\PCA data rotated.png)

### what the PCA have done?  An explanation. 



the `pcaTr`  from above have rotation matrix and its corresponding explained variance which can be accessed using

1. `pcaTr.components_` : has the rotation matrix, 
2. `pcaTr.explained_variance_` has the explained variance of each principal component



```python
print('Eigenvectors or principal component: First row must be in the direction of [1, n]')
print(pcaTr.components_)

## output
#Eigenvectors or principal component: First row must be in the direction of [1, n]  
# [[ 0.70710678  0.70710678]
# [ 0.70710678 -0.70710678]]
```



```python
print('Eigenvalues or explained variance')
print(pcaTr.explained_variance_)

# output
Eigenvalues or explained variance
[1.57423216e-01 8.85544116e-34]
~ [ 0.166, 0 ]
```

now, rotation matrix has the form 
$$
R = \begin{bmatrix} cos(45^o) & sin(45^o) \\ -sin(45^o) & cos(45^o) \end{bmatrix}
$$
**We made two features x, y with y = n * x. The PCA has identified the relation as rotation and remove the data features redundancy.** 



The explained variance or eigen values [ 0.166, 0 ]. Now if we calculate the [variance of uniform random variable](https://revisionmaths.com/advanced-level-maths-revision/statistics/uniform-distribution) 
$$
Var(x) = \frac {(b - a)^2}{12}
$$

$$
 = \frac { (1-2)^2 }{12} = 0.08333
$$

Then the explained variance given by the PCA can be interpret as

$$[Var(x) + Var(y)  \ 0] = [0.0833 + 0.0833 \  0] = [0.166 \ 0]$$ 

which means all the explained variance of our new system is explained by our first principal component. 



------

## Code 2 : Correlated Normal Random Variables

Now, sample the x and y as 2 random variable with different variances and with a specific covariance b/w them. 

1. Create two independent normal random variables with desired variances
2. combine them using a rotation matrix 

In this way the new resulting variables will be linear combination of the original random variables and thus be dependent and correlated.

```python
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms  
import random 

random.seed(100)
std1 = 1; std2 = 0.333 # the desired deviation of our random variables

x = np.random.normal( loc = 0, scale = std1, size = 1000 ) # mean, std, size
y = np.random.normal(loc = 0, scale = std2, size = 1000 ) 

x = x - np.mean(x) # Center x 
y = y - np.mean(y) # Center y 
```

```
plt.scatter(x, y)
plt.show() 
```

![pca data 2](F:\Github\NLP\Notes\Images\pca data 2.png)

```python
#Define a pair of dependent variables with a desired amount of covariance
n = 1 # Magnitude of covariance. 
angle = np.arctan(1 / n) # Convert the covariance to and angle
print('angle: ',  angle * 180 / math.pi) 

# output 
# angle = 45.0 

# Create a rotation matrix using the given angle
rotationMatrix = np.array([[np.cos(angle), np.sin(angle)],
                 [-np.sin(angle), np.cos(angle)]])
print('rotationMatrix')
print(rotationMatrix)

#output
#rotationMatrix
##[[ 0.70710678  0.70710678]
 #[-0.70710678  0.70710678]]
    
xy = np.concatenate(([x] , [y]), axis=0).T
data = np.dot(xy, rotationMatrix)
plt.scatter(data[:,0], data[:,1])
plt.show()
```

![pca data 2 rotated](F:\Github\NLP\Notes\Images\pca data 2 rotated.png)



 Now we will use PCA to get the Principal components. 

**Note:** In theory, the Eigenvector matrix must be the inverse of rotation matrix



```python
pca = PCA( n_components = 2 )
pcaTr = pca.fit(data) 
dataPCA = pcaTr.transform(data) 

print(pcaTr.components_) 
# output
# [[-0.707714   -0.70649904]
# [-0.70649904  0.707714  ]]

print(pcaTr.explained_variance_)
# [0.95925777 0.11811936]
# [ ~1, ...]
```

```python
plt.scatter(data[:,0], data[:,1]) 
plt.scatter(dataPCA[:,0], dataPCA[:,1])

# Plot the first component axe. Use the explained variance to scale the vector
plt.plot([0, rotationMatrix[0][0] * std1 * 3], [0, rotationMatrix[0][1] * std1 * 3], 'k-', color='red')
# Plot the second component axe. Use the explained variance to scale the vector
plt.plot([0, rotationMatrix[1][0] * std2 * 3], [0, rotationMatrix[1][1] * std2 * 3], 'k-', color='green')

plt.show()
```

![ pca data result ](F:\Github\NLP\Notes\Images\pca data 2 final.png)
**Explanation :** 

1. The rotation matrix used to create our correlated variables took the original uncorrelated variables `x` and `y` and transformed them into the blue points.
2. The PCA transformation finds out the rotation matrix used to create our correlated variables (blue points). Using the PCA model to transform our data, puts back the variables as our original uncorrelated variables.
3. The explained Variance of the PCA is [ 1.0094, 0.1125  ] which is approximately $$[1, 0.333 * 0.333] = [std1^2, std2^2],$$



which is approximately



[1,0.333∗0.333]=[𝑠𝑡𝑑12,𝑠𝑡𝑑22],

------

## Code 3 : PCA as a strategy for dimensionality reduction

The principal components contained in the rotation matrix, are decreasingly sorted depending on its explained variance. It usually means that the first components retain most of the power of the data to explain the patterns that generalize the data. 

*However for tasks like **novelty detection**, we are interested in the patterns that explain much less variance.* 



Now  for high dimension data, we can reduce it to two dimensions. In that reduced space of uncorrelated variables, we can easily separate e.g cats and dogs.

for example, images of cats and dogs. A visualization is ![catdog ](F:\Github\NLP\Notes\Images\catdog.png)

------

### Reducing the dimension  of word embeddings for visualization : *A hands-on implementation of PCA from scratch* 

We have google word vector embeddings of 300 dim. Whereas it's easy to work with the embeddings as the follow rules of algebra, it's not feasible to visualize them. 

*You can think of PCA as a method that projects our vectors in a space of reduced dimension, while keeping the maximum information about the original vectors in their reduced counterparts.* 

In this case, by ***maximum information*** we mean that the Euclidean distance between the original vectors and their projected siblings is minimal. Hence vectors that were originally close in the embeddings dictionary, will produce lower dimensional vectors that are still close to each other.

If you remember the steps from above.

1. *Mean normalize the data*

2. *Compute the covariance matrix of your data*

3. *Compute the eigenvectors and the eigenvalues of your covariance matrix, sorted by decreasing order of their eigen values*

4. *Multiply the first K eigenvectors by your normalized data.*

   ![It should look like this ](F:\Github\NLP\Notes\Images\word_embf.jpg)

   ```python
   def compute_pca(X, n_components=2):
       
       X_demeaned = X - np.mean ( X, axis = 0 )
   
       # calculate the covariance matrix
       covariance_matrix = np.cov ( X, rowvar=False )
   
       # calculate eigenvectors & eigenvalues of the covariance matrix
       eigen_vals, eigen_vecs = np.linalg.eigh ( covariance_matrix, UPLO='L')
   
       # sort eigenvalue in increasing order (get the indices from the sort)
       idx_sorted = np.argsort( eigen_vals )
       
       # reverse the order so that it's from highest to lowest.
       idx_sorted_decreasing = idx_sorted[::-1]
   
       # sort the eigen values by idx_sorted_decreasing
       eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]
   
       # sort eigenvectors using the idx_sorted_decreasing indices
       eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing]
   
       # select the first n eigenvectors (n is desired dimension
       # of rescaled data array, or dims_rescaled_data)
       eigen_vecs_subset = eigen_vecs_sorted[:, 0:n_components]
   
       # transform the data by multiplying the transpose of the eigenvectors 
       # with the transpose of the de-meaned data
       # Then take the transpose of that product.
       X_reduced = np.dot( eigen_vecs_subset.T, X_demeaned.T).T
   
       return X_reduced
   
   ```

   By using PCA over the word embeddings, you can get results as shown. 

   ![cluster](F:\Github\NLP\Notes\Images\clusters.png)

------

## Resources

1. Coursera NLP Specialization 
2. Eigen values and Eigen vectors
3. Linear Algebra
4. A good video on PCA. 