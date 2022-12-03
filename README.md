# DeepFL

## 1. About project
Model takes images from path and evaluates content and technical parameters, returns 0 or 1.  

<p align="center">
  <img src="project_images/project.png" style='width:60%; background-color: white'>
</p>

Developing an AI which will predict if the input image is aesthetic or not.

    -->  Creating multi-model AI, which prediction depends on quality and content of input images 
    -->  Using, developing and optimizing feature extractor, which responsible for input images aestheticness
    -->  Trying various ML (for ex. dimensionality reduction) techniques for extracted features transformation
    -->  Creating a Fully Connected neural network for image classification (aesthetic or not)
    -->  Analyzing and cleaning training data (80k), and big test data (Big Data problems)
    -->  Using knowledge distillation for optimizing feature extractor and FC
    -->  Using Explainable AI techniques (SHAP, LIME, LRP) to get explanations of predictions
    -->  Data augmentation techniques on training dataset for making sensitive model, and on testing dataset to check sensitivity of model
    -->  Building an user friendly folder architecture and code, used OOP structure
    Qualifications:
    Neural Networks (CNN, Fully-Connected, Feature extractor), Machine Learning,
    Supervised Learning, Statistics, Knowledge distillation, Explainable AI,
    Dimensionality Reduction, PCA, ICA, Tensorflow, Python, OOP,  Numpy, Big Data
    Data Augmentation, Git, OpenCV, Jupyter Notebook
____
____

## 2. Best result

<table style="width:100%">
  <tr>
    <th>Benchmark1</th>
    <th>Benchmark2</th>
    <th>Alm Benchmark</th>
    <th>Custom Benchmark</th>
    <th>Coco Benchmark</th>
  </tr>
  <tr>
    <td>90.9%</td>
    <td>90.5%</td>
    <td>94%</td>
    <td>94%</td>
    <td>94.6%</td>
  </tr>
</table>

_____
_____

## 3. Data description
Training data contains 70075 bad images and 11024 good images

Test data:
   - Benchmark1: 111
      
         --> 47 aesthetic      
         --> 64 not aesthetic   
   - Benchmark2: 1034

         --> 534 aesthetic
         --> 500 not aesthetic   

   - Alm benchmark:        200 

         --> 100 aesthetic
         --> 100 not aesthetic

   - Custom benchmark:     200 

         --> 100 aesthetic
         --> 100 not aesthetic

   - COCO benchmark:       1000

         --> 0 aesthetic
         --> 1000 not aesthetic

___
---

## 4. Model

###Model architecture and hyper-parameters were chosen based on the performance on the validation set.

At first the network  resizes input image (max_resize 996x996) and gives resized image to the MultiGap network , 
in parallel of this, resizes input image with borders (resize_with _borders 600x600) and gives resized image to 
the CNN( EfficientNetB7).MultiGap returns a vector of dimensions 1x16928, after net reduces dimensions with PCA 
to 1x8464. Net  does the same with CNN’s feature vector (1x2560 → 1x1280),then concatenates two vectors
(dimension will be 1x9744) and gives it to the Fully Connected network , which will return prediction.


![App Screenshot](project_images/diagram.png)

----
---
### 4.1 MultiGap 

MultiGap(Multi-pooled inception network)

MultiGap is a new approach to image aesthetic evaluation. 

       1) a novel deep neural network architecture called MultiGAP that exploits features from multiple inception 
          modules pooled by global average pooling (GAP), evaluated by prediction of (10-class) categorical distribution, before performing score binarization; 

       2) the incorporation of textual features trained simultaneously with MultiGAP using a recurrent neural 
          network (RNN) with gated recurrent unit (GRU) layers;

       3) the ability to leverage the GAP block for visualization of
          activation maps based on the aesthetic category and rating.

On top of that, an extensive performance comparison between
competing methods in literature is also reported for the benefit of this growing field of research.

To learn visual features, are used GoogLeNet, a deep neural network architecture codenamed Inception proposed by Google 
as our base network. Briefly, the GoogLeNet is a network consisting of inception modules stacked upon each other, 
with occasional max-pooling layers with stride 2 added to halve the grid resolution. A single inception module consists
of a number of parallel filters to densely capture spatial correlations. Dimensionality reduction is incorporated 
through the use of 1 × 1 convolutions before the expensive 3 × 3 and 5 × 5 convolutions(see at the figure below)


<p align="center">
  <img src="project_images/Inception_module.png" style='width:50%; background-color: white'>
</p>

To construct network MultiGAP, the final three inception modules are removed and a new “GAP block” is attached to each 
inception module. Each GAP block consists of an additional convolution layer followed by a global average pooling (GAP) 
layer, which reduces the feature maps to a 1-D vector of weights (see at the figure below). 

<p align="center">
  <img src="project_images/Gap_block.png" style='width:50%; background-color: white'>
</p>

GAP layers are useful replacements for fully connected layers.
By adding GAP layers to the inception modules, this intuitively regularizes the ”micro network” within each module, 
reducing the proneness to overfitting at different levels of abstraction.
Then replaced the original 1000-class softmax (of ImageNet Challenge) with a 10-class softmax.

To get an IQA, the process has two steps. The first step uses multi-task learning (MTL) to predict the objective 
quality scores of multiple FR-IQA metrics simultaneously. Introduced histogram equalization used to improve learning. 
The MTL framework allows you to train a deep model on a large-scale unrated image set, such as KADIS-700k. 
In the second step, Multi-Level Spatially Pooled (MLSP) includes the extracted from the finely tuned CNN model 
from the first step to represent the given image. For IQA database data, e.g. KADID-10k, a new regression network is trained
non-linearly map MLSP features to their respective subjective indicators of quality.

Because the objective quality measures of FR-IQA(Full-Reference Image Quality Assessmen) methods are highly correlated 
with subjective quality measures, the learned function is also expected to be highly correlated with subjective 
quality measures. To this end, multitasking training with deep CNNs is performed to model the described function.

The architecture deep CNN model for MTL(Multi-Task Learning) is illustrated in below figure. We feed an image into the CNN
base of InceptionResNetV2 and use Global Average Pooling (GAP) for each feature map. The resulting feature vector passes
through K separate task modules, predicting the K FR-IQA scores.
Each task module consists of five layers. These are fully-connected (FC) with  512 units, dropout with rate 0.25,
FC with 256 units, dropout with rate 0.5, and output with one neuron. The deep CNN model aims to predict the objective 
scores from K = 11 available FR-IQA methods. These methods are SSIM, MS-SSIM, IW-SSIM, MDSI, VSI, FSIMc, GMSD, SFF, 
SCQI, ADD-GSIM and SR-SIM. The MTL framework has the following merits. Firstly, it is very convenient to generate 
distorted images given large-scale pristine reference images and to estimate their objective quality scores
via FR-IQA methods. Therefore, training a deep CNN on large-scale images to address over-fitting becomes feasible.
Secondly, although several FR-IQA methods have been proposed in the last decade, there is no single FR-IQA method 
that significantly outperforms the rest. One method may be superior for some distortion types, but inferior for 
others as different methods extract different features and use different pooling strategies. Therefore, learning to 
predict objective quality scores from multiple FR-IQA methods rather than from a single FR-IQA method by using a shared 
CNN body reduces the risk of over-fitting.

<p align="center">
  <img src="project_images/FR_IQA.png" style='width:60%; background-color: white'>
</p>


To make a fair and comprehensive comparison, were conducted as follows: In the MTL stage, KADIS700k was divided 
into a training set of 600,000 images and a validation set of 100,000 images. There was no test set as it was 
unnecessary to evaluate the model at this stage. In the visual quality prediction stage, all databases were randomly 
split into a training set (60%), a validation set (20%), and a test set (20%). All the divisions are on the basis of 
the content of reference images so there is no content overlapping between the three subsets.
All the models and loss functions in experiments were implemented using the Python Keras library with Tensorflow as a
backend  and ran on two NVIDIA Titan Xp GPUs with a batch size of 64. The InceptionResNetV2 CNN body was initialized 
with pre-trained weights from ImageNet. The weights of FC layers were initialized with the method of He et al. 
The ADAM optimizer was used for optimization, with a custom learning rate α and default parameters β1 = 0.9, β2 = 0.999. 
In the MTL stage, fixed α at 10^(−4) . In the visual quality prediction stage, use α = 10^(−1) , 10^(−2) , . . . , 10^(−5) and 
chose the value that gave the best validation result. Considering the large size of KADIS-700k, 
only trained for 10 epochs in the MTL stage. Trained for 30 epochs in the visual quality prediction stage.

----
----
### 4.2 CNN
In project has been used EfficientNetB7 as CNN.

EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of 
depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, 
the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients.

-----------------

<p align="center">
<font size="+3">
EfficientNet-B7 architecture  
</font>
</p>

<p align="center">
  <img src="project_images/EfficientNetB7.png" style='width:70%; background-color: white'>
</p>

EfficientNet-B7 network has a 64.097.680 parameters. <br>
The model accepts inputs of any size.

<p align="center">
<font size="+2">
MBconv1 and MBconv6 architecture
</font>
</p>


<p align="center">
  <img src="project_images/MBNet_conv.png" style='width:50%; background-color: white'>
</p>

EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet,  while being 8.4x smaller and 6.1x faster
on inference than  the best existing ConvNet. EfficientNets also transfer well and achieve state-of-the-art accuracy 
on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. 

----
----

### 4.3 FC 

The Fully Connected network consists of 3 dense layers, with batch norm and dropout layers in between. It takes 
a concatenated feature vector as an input, and outputs a probability score for each of the 2 labels. 


<p align="center">
  <img src="project_images/FC.png" style='width:50%; background-color: white'>
</p>

----
----

## 5. Training

Model training:

    --> Extracting train image multigap features and saving with .json

    --> Extracting train image cnn features and saving

    --> Transformed multigap and cnn features. 

    --> Loading cnn features and multigap features from .json. Concatenate  and train with FC


----
____
### Usage

### 1. Principal Component Analysis (PCA)
The goal of principal component analysis is to compute the most meaningful
basis to re-express a data set. X is the original data set, where each column is
a single sample of our data set. In the case here X is an m x n matrix where m
is the number of features and n is the number of data points. Let Y be another
m x n matrix composed by applying a linear transformation P on our X. X is
the original recorded data set and Y is a re-representation of that data set


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\inline&space;\tiny&space;PX&space;=&space;Y&space;" style='width: 200px; height: 40px; background-color: white; '>
  
  The rows of P are a new set of basis vectors for representing columns of X.
The row vectors {
    <img src="https://latex.codecogs.com/svg.image?\inline&space;\tiny&space;\bg{white}p_1,&space;.&space;.&space;.&space;,&space;p_m&space;" style='width: 55px; height: 9px; background-color: white; '>
} in the transformation P will become the principal
components of X.
  Another thing to know is covariance.
Covariance measures the degree of
the linear relationship between two variables. We define the covariance matrix
C. which is a symmetric m x m matrix, representing the covariance between
features. The diagonal terms of C are the variance of particular features on the
span of our data.
The off - diagonal terms of C are the covariances between
features. Thus, we want our C to be diagonal, because we want the covariance
between features to be 0.
We also want the variances on the diagonal to be
maximized.

We define a new matrix
<img src="https://latex.codecogs.com/svg.image?\inline&space;\tiny&space;\bg{white}&space;A&space;\equiv&space;&space;XX^{T}" style='width: 80px; height: 8px; background-color: white; '>
(the covariance matrix for our X data) where A is symmetric.
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\inline \bg{white}C_Y = YY^{T} = (PX)(PX)^{T} = PXX^{T}P^{T} = P(XX^{T})P^{T} = PAP^{T} \" style='width: 60%; background-color: white; '>

  A symmetric matrix (A) is diagonalized by an orthogonal matrix of its eigenvectors. 
The matrix P is selected to be a matrix where each row 
  <img src="https://latex.codecogs.com/svg.image?\inline&space;\tiny&space;\bg{white}p_i" style='width: 1.5%; background-color: white; '>
 is an eigenvector of 
 <img src="https://latex.codecogs.com/svg.image?\inline&space;\tiny&space;\bg{white}&space;XX^{T}" style='width: 3%; background-color: white; '>
. So 
<img src="https://latex.codecogs.com/svg.image?\inline&space;\tiny&space;\bg{white}&space;P&space;=&space;&space;E^{T}" style='width: 5%; background-color: white; '>
</p>

<p align="center">
 <img src="https://latex.codecogs.com/svg.image?\inline&space;\tiny&space;A&space;=&space;EDE^{T}&space;" style='width: 300px; height:40px; background-color: white; '>
</p>

Where D is the Diagonal matrix (which will be the covariance matrix of the output Y).
 Thus, we redefine 
  <img src="https://latex.codecogs.com/svg.image?\inline&space;\tiny&space;A&space;=&space;P^{T}DP&space;" style='width: 7%; background-color: white; '>
to
  <img src="https://latex.codecogs.com/svg.image?\inline&space;\tiny&space;EDE^{T}&space;" style='width: 4%; background-color: white; '>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\inline&space;\tiny&space;\bg{white}C_Y&space;=&space;PAP^{T}&space;=&space;P(P^{T}DP)P^{T}=(PP^{T})D(PP^{T})&space;=&space;(PP^{-1})D(PP^{-1})&space;=&space;D" style='width: 1000px; height: 40px; background-color: white; '>

So the diagonal matrix D is computed, which represents the covariance of our Y output. 
That means that Y has 0 covariance between features and maximized variance for each feature. 
For dimensionality reduction the selected P transform matrix contains only a desired number of 
components (eigenvectors with larger (sorted) eigenvalues). After that the XP dot product 
is computed and results in Y output..  	
In our case 8464 was selected to be the number of components to reduce MultiGap features,
 and 1280 to reduce CNN features. Each of the 2 PCA models was separetly fitted on ~80000 
training data points (16928x80000 matrix extracted from MultiGap and 2560x80000 matrix 
extracted from CNN). After the fitting process the models were saved. Now, during each 
prediction the feature vector is reduced by passing through the corresponding model 
(pca_cnn or pca_mg), and only after that enters the fully connected part (which was trained 
on pca-reduced feature vectors for the 80000 data points (16928 -->8464 for MG and 2560 -->1280 for CNN).
</p>

### 2. XAI

Explainable artificial intelligence (XAI) is a set of processes and methods that allows human
users to comprehend and trust the results and output created by machine learning algorithms.
Explainable AI is used to describe an AI model, its expected impact and potential biases.
It helps characterize model accuracy, fairness, transparency and outcomes in AI-powered decision making.
Explainable AI is crucial for an organization in building trust and confidence when putting 
AI models into production. AI explainability also helps an organization adopt a responsible approach to AI development.

<p align="center">
  <img src="project_images/XAI.png" style='width:60%; background-color: white'>
</p>

The Explainable AI (XAI) program aims to create a suite of machine learning techniques that:

    1.Produce more explainable models, while maintaining a high level of learning performance (prediction accuracy)
    2.Enable human users to understand, appropriately trust, and effectively manage the emerging generation of artificially
    intelligent partners.
----

#### 2.1 Shap

SHAP (Shapley Additive Explanations) by Lundberg and Lee (2016) is a method to explain individual
predictions, based on the game theoretically optimal Shapley values. 
The goal of SHAP is to explain the prediction of an instance x by computing the contribution of each feature 
to the prediction. The SHAP explanation method computes Shapley values from coalitional game theory. 
The feature values of a data instance act as players in a coalition. Shapley values tell us how to fairly 
distribute the “payout” (= the prediction) among the features. A player can be an individual feature value, 
e.g. for tabular data. A player can also be a group of feature values. For example to explain an image, pixels
can be grouped to superpixels and the prediction distributed among them. One innovation that SHAP brings to the 
table is that the Shapley value explanation is represented as an additive feature attribution method, a linear model.
That view connects LIME and Shapley values. SHAP specifies the explanation as


<p align="center">
 <img src="https://latex.codecogs.com/svg.image?g({z}')&space;=&space;\phi&space;_{0}&space;&plus;&space;\sum_{j=1}^{M}\phi&space;_{j}{z}'_{j}" style='width: 30%; background-color: white; '>

where g is the explanation model, 
<img src="https://latex.codecogs.com/svg.image?{z}'&space;\epsilon&space;&space;\begin{Bmatrix}0&space;&&space;&space;1\\\end{Bmatrix}^{M}" style='width: 7%; background-color: white; '>
is the coalition vector, M is the maximum coalition size and  
<img src="https://latex.codecogs.com/svg.image?\phi_{j}&space;\epsilon&space;R" style='width: 4%; background-color: white; '>
is the feature attribution for a feature j, the Shapley values.
</p>


<p align="center">
  <img src="project_images/Shap.png" style='width:60%; background-color: white'>
</p>

In this project, Shap explains why the model made such a prediction (aesthetic or not aesthetic).
The top figure shows the result of Shap. The red ones are the segments that contributed to the prediction,
and the blue ones counteracted to the prediction.

-------
#### 2.2 Lime

The beauty of LIME(Local Interpretable Model-agnostic Explanations) its accessibility and simplicity. 
The core idea behind LIME though exhaustive is really intuitive and simple.

    1.Model agnosticism refers to the property of LIME using which it can give explanations for any 
      given supervised learning model by treating as a ‘black-box’ separately. This means that LIME can 
      handle almost any model that exists out there in the wild!
    2.Local explanations mean that LIME gives explanations that are locally faithful within the 
      surroundings or vicinity of the observation/sample being explained.


When given a prediction model and a test sample, LIME does the following steps:

    1.Sampling and obtaining a surrogate dataset: LIME provides locally faithful explanations around
      the vicinity of the instance being explained. By default, it produces 5000 samples(in this project num_samples=1000)
      of the feature vector following the normal distribution. Then it obtains the target variable for these 
      5000 samples using the prediction model, whose decisions it’s trying to explain.
    2.Feature Selection from the surrogate dataset: After obtaining the surrogate dataset, it weighs
      each segment according to how close they are from the original sample/observation. Then it uses a feature
      selection technique like Lasso to obtain the top important features.

<p align="center">
  <img src="project_images/Lime.png" style='width:60%; background-color: white'>
</p>

In the top figure the green ones are the segments that contributed to the prediction,
and the red ones counteracted to the prediction.

-----
-----

### 3. Genetic Algorithm
Genetic algorithms are a particular class of evolutionary algorithms (also known as evolutionary computation) 
that use techniques inspired by evolutionary biology such as inheritance, mutation, selection, and crossover 
(also called recombination). It is based on Charles Darwin's theory of natural evolution and uses the principles
of selection and evolution to produce several solutions to a given problem.

<p align="center">
  <img src="project_images/GA_structure.png" style='width:40%; background-color: white'>
</p>

Genetic algorithms terminology:

    Individual - Any possible solution
    Population - Group of individuals
    Search Space - All possible solutions
    Chromosome - Blueprint for an individual
    Trait - Possible aspect of an individual
    Allele - Possible settings for a trait
    Locus - The position of a gene on the chromosome
    Genome - Collection of all chromosomes for an individual

#### Input

The input to the GA is a set of potential solutions to that problem, encoded in some fashion. These candidates may be 
solutions already known to work, with the aim of the GA being to improve them, but more often they are generated at random.

#### Fitness function

A metric that allows each candidate to be quantitatively evaluated.

#### Summary

The GA evaluates each candidate according to the fitness function activity.
Individuals with high fitness can be termed as promising candidates. These
promising candidates are kept and allowed to reproduce. From them multiple
copies are made, but the copies are not perfect; random changes are introduced
during the copying process. These digital offspring then go on to the next
generation, forming a new pool of candidate solutions, and are subjected to a
second round of fitness evaluation. Those candidate solutions which were
worsened, or made no better, by the changes to their code are again deleted. The
expectation is that the average fitness of the population will increase each round.

<p align="center">
  <img src="project_images/GA_components.png" style='width:40%; background-color: white'>
</p>

#### Population initialization

Population Initialization in Genetic Algorithms

Population Initialization is the first step in the Genetic Algorithm Process. Population is a subset of solutions 
in the current generation. Population P can also be defined as a set of chromosomes. The initial population P(0), 
which is the first generation is usually created randomly. In an iterative process, populations P(t) at generation 
t (t =1,2,....) are constituted. When dealing with genetic algorithms, the diversity of the population should be
maintained otherwise it might lead to convergence of algorithms before reaching a global optimum.


### Generation Gap

It is a special term which varies between 0<=G<=1 . It is defined as the proportion of chromosomes in the population
which are replaced in each generation.(It is the ratio of children of individuals in the previous generation to the 
population size). If G=0.9, it means that 90% chromosomes from the old population are replaced by 90% best found 
chromosomes from the new population while preserving 10% best found chromosomes from the old population into the 
new population. When G increases global search capability increases while local search capability decreases.

-----------

### Selection

<p align="center">
  <img src="project_images/selection.png" style='width:60%; background-color: white'>
</p>

#### Fitness Proportionate Selection

In this method, an individual can become a parent with a probability which is proportional to its fitness. 
Therefore, fitter individuals have a higher chance of reproduction and propagating their features to the next
generation.Thus this applies a selection pressure to the more fit individuals in the population, evolving better 
individuals over time.Two implementations of fitness proportionate selection are possible−Roulette Wheel Selection
and Stochastic Universal Sampling.

#### Roulette Wheel Selection

Selection in this method is proportionate to the fitness of an individual. Higher the
fitness of an individual,higher the chances of getting selected.(Higher the fitness
greater the pie on the wheel and therefore a greater chance of landing in front of
the fixed point when the wheel is rotated.)

<p align="center">
  <img src="project_images/roulette.png" style='width:40%; background-color: white'>
</p>

For the implementation of the algorithm we get probabilities of each chromosome
which are equal to fitness of each chromosome divided by the sum of all fitnesses.
Then we pick chromosomes according to their probability. This process is repeated
until the desired number of individuals is selected.


----------------------

#### Ordinal based methods

#### Ranking selection

Roulette selection will have problems when the fitness differs very much. For
example, if the best chromosome fitness is 90% of all the roulette wheel then other
chromosomes will have very few chances to be selected.
Rank selection first ranks the population and then every chromosome receives
fitness from this ranking. The worst will have fitness 1, second 2 etc.

<p align="center">
  <img src="project_images/ranking.png" style='width:60%; background-color: white'>
</p>


Advantages:The advantage of this method is that it can prevent very fit individuals
from gaining control early at the expense of less fit ones, which would reduce the
population's genetic diversity and might block attempts to find an acceptable
solution.

Disadvantages: The disadvantage of this method is that it required sorting the
entire population by rank which is a time consuming procedure

------------------
#### Threshold based

#### Truncation selection
It is an artificial selection method which is used by parents for large populations/mass selection.
Truncation selection, individuals are sorted according to their fitness. Only the best
individuals are selected as parents. Truncation threshold Trunc is used as the parameter for truncation selection.
Trunc indicates the proportion of the population to be selected as parents and takes values
ranging from 50%-10%. Individuals below the truncation threshold do not produce offsprings.

You can see other selection methods in this link
https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm

------------------

### Crossover

The crossover operator is analogous to reproduction and biological crossover. In this more than one parent is selected
and one or more off-springs are produced using the genetic material of the parents. Crossover is usually applied in a
GA with a high probability.

<p align="center">
  <img src="project_images/crossover.png" style='width:60%; background-color: white'>
</p>

#### Single-Point Crossover
 
Single-Point Crossover - Select a crossover point at random and
swap the bits at the right site

<p align="center">
  <img src="project_images/single_point.png" style='width:40%; background-color: white'>
</p>

#### Two-point Crossover

Two-Point Crossover - Select a crossover points at random and
swap the bits at the middle site.

<p align="center">
  <img src="project_images/two_point.png" style='width:40%; background-color: white'>
</p>

#### Multi-Point Crossover

Multi-Point Crossover - Select multiple crossover points at random
and swap the bits at the alternate sites.

<p align="center">
  <img src="project_images/multi_point.png" style='width:40%; background-color: white'>
</p>

You can see other crossover methods in this link
https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm

-----

### Mutation

Mutation may be defined as a small random tweak in the chromosome, to get a new solution.
It is used to maintain and introduce diversity in the genetic population.

#### Bit Flip Mutation

In this bit flip mutation, we select one or more random bits and flip them.

<p align="center">
  <img src="project_images/bit_flip.png" style='width:60%; background-color: white'>
</p>

#### Random Resetting

Random Resetting is an extension of the bit flip for the integer representation. In this, a
random value from the set of permissible values is assigned to a randomly chosen
gene.

#### Swap Mutation

In swap mutation, we select two positions on the chromosome at random, and interchange the values.

<p align="center">
  <img src="project_images/swap.png" style='width:60%; background-color: white'>
</p>

#### Scramble Mutation

Scramble mutation is also popular with permutation representations. In this, from the
entire chromosome, a subset of genes is chosen and their values are scrambled or shuffled.

<p align="center">
  <img src="project_images/scramble.png" style='width:60%; background-color: white'>
</p>

#### Inversion Mutation

In inversion mutation, we select a subset of genes like in scramble mutation, but
instead of shuffling the subset, we merely invert the entire string in the subset.

<p align="center">
  <img src="project_images/inversion.png" style='width:60%; background-color: white'>
</p>

#### Center inverse mutation

The chromosome is divided into two sections. All genes in each section are copied
and then inversely placed in the same section of a child.

#### Thrors Mutation

Three genes are chosen randomly which shall take the different positions i < j < l.
The gene of the position i becomes in the position j and the one who was at this
position will take the position l and the gene that has held this position takes the
position i.

----------

### Survival Selection

The Survivor Selection Policy determines which individuals are to be kicked out
and which are to be kept in the next generation. It is crucial as it should ensure that
the fitter individuals are not kicked out of the population, while at the same time
diversity should be maintained in the population.

#### Age Based selection

In Age-Based Selection, we don’t have a notion of fitness. It is based on the
premise that each individual is allowed in the population for a finite
generation where it is allowed to reproduce, after that, it is kicked out of the
population no matter how good its fitness is.

#### Fitness based selection

In this fitness based selection, the children tend to replace the least fit
individuals in the population. The selection of the least fit individuals may
be done using a variation of any of the selection policies – tournament
selection, fitness proportionate selection, etc.

------------------
------------------
