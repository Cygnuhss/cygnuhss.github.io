---
layout: post
title: Neural Networks - Part 1:<br>Single-layer Perceptron
---

Neural networks are **classifiers**, meaning that they can be used to determine
which class an object belongs to. There are several types of neural networks,
such as the multilayer perceptron, which will be discussed in [subsequent posts]({% post_url 2015-08-26-Multilayer-Perceptron %}).
Here, I will explain a simple kind of neural network, the single-layer
perceptron.

## Classification
To understand the goal of neural networks as classifiers, we have to consider
the following situation.


You are an extraterrestrial form of life that is on Earth to research human
life. You notice that the planet's inhabitants are eating fruits that they call
`apple` and `banana`. These are **classes** of fruit. When a human is eating a
fruit, it belongs either to the class `apple`, or to the class `banana` (for
the sake of argument, these are the only fruit humans dare to eat).

{% include image.html img="img/single-layer-perceptron/apple+banana.png" title="Apple and Banana" caption="The classes Apple and Banana are the only fruit humans eat." %}


Your research consists of tallying the fruits humans are eating. Before you
can tally an occurrence of a fruit, you have to know which class it belongs to.
Unfortunately, when you look at the fruit, you cannot discern between the two
classes. You can write down `apple`, having a 50% chance of getting it right
(assuming humans eat as many apples as bananas), but that does not seem very
accurate.


You can, however, make the person do the classification for you by
asking him what fruit he is eating (humans do not make mistakes in classifying
apples and bananas). This will take a long time, as humans are extremely slow
to respond compared to your optimised alien mind, so this does not seem like a
reasonable strategy. All hope is not lost, because you notice three helpful
things:

1. Humans can **perfectly classify** apples and bananas.
2. All fruits tend to be on a **colour scale**, ranging from red to yellow.
3. All fruits tend to be on a **shape scale**, ranging from circle-like to
line-like.

Based on the last two rules, every fruit can be placed somewhere in a graph
with two axes, one for colour and one for shape. The two scales are
**dimensions** of this classifier.


As an alien, you do not know this yet, but humans define `apple` as something
that lies low on the colour axis (it is a red object, rather than yellow), and
it also lies low on the shape axis (it is a circle-like object, rather than a
line-like object). A `banana`, on the other hand, lies high on the colour axis
(it is yellow, rather than red) and high on the shape axis (it is line-like,
rather than circle-like).

{% include image.html img="img/single-layer-perceptron/colour+shape.png" title="Colour and Shape scales" caption="The classes Apple and Banana can be placed along a colour and shape scale." %}


These two definitions of the classes will split up the space with a line.
Everything on one side of the line is an `apple`, because it is more red and/or
circle-like. Everything on the other side of the line is a `banana`, because it
is more yellow and/or line-like.

{% include image.html img="img/single-layer-perceptron/classification.png" title="Classification" caption="All the apples are under the line, all the bananas above it." %}


You resolve to devise a system to learn to classify apples and bananas
yourself, so that you do not need to rely on human classification any longer.
This system will be a **single-layer perceptron** specialised in deciding whether
a fruit is of the class `apple`, or of the class `banana`.

{% include image.html img="img/single-layer-perceptron/devise-classification.png" title="Devise classification" caption="Which line splits Apple and Banana best? A perceptron will help decide." %}

## Network
A single-layer perceptron is a **feedforward neural network**, which means that
the flow of information is forward through the network. This is unlike a
**recurrent neural network**, where cyclical flows of information exist.


A single-layer perceptron, as the name implies, consists of a single layer of
**nodes**, which will also be the output layer. A node is a machine that gets
inputs and applies an **activation function** to it to return an output.
The inputs would be the degree of colour and the degree of shape in our case
of apple-banana classification, both real numbers.

{% include image.html img="img/single-layer-perceptron/perceptron-layers.png" title="Perceptron layers" caption="The perceptron consists of a single node, getting two inputs and returning one output." %}

The input, in other words, can be described as a real-valued vector:

$$x = (I_{colour},I_{shape})$$

Or more generally:

$$x = (I_1,I_2,\dotsc,I_n)$$


The activation function answers yes/no questions by either activating (I will
give this the value \\( 1 \\)), or deactivating (this will have the value
\\( 0 \\)).  
This process of either firing or not firing is similar to how neurons in the human
brain work, hence the name *neural* networks.

The classification of apples and bananas can be described as a yes/no question:

**Is it red and circle-like?**  
Yes \\( \longrightarrow \\) It is an `apple`.  
No  \\( \longrightarrow \\) It is a `banana`.

{% include image.html img="img/single-layer-perceptron/binary-tree.png" title="Binary tree" caption="If the object is red and line-like, it is an apple." %}

The yes-variant follows from the definition of an `apple`. The no-variant
follows from not being an `apple`, and there are only apples and bananas.
Likewise a banana-question could be formed, where *Yes* would result in
`banana` and *No* in `apple`. Both questions classify apples and bananas
correctly.


The activation function for the apple-question is thus a function that returns
\\( 1 \\) if the inputs are reddish and circle-like, and \\( 0 \\) if the inputs
are yellowish and line-like.  
Both colour and shape might not be equally good predictors of class. It might be
the case that colour is not as good a predictor (because some apples are slightly
yellow), and shape turns out to be more important when classifying apples and
bananas.


This is where **weights** come in. Each input has a certain weight, that states
how important that input is. A more important input will influence the activation
function more than a less relevant input. An object can be nearing yellow, yet
still be classified as an `apple` if it is round.

{% include image.html img="img/single-layer-perceptron/perceptron-weights.png" title="Weights" caption="Inputs can have different weights, or importance." %}

The weights are written down as a real-valued vector:

$$w = (w_{colour},w_{shape})$$

Or more generally:

$$w = (w_1,w_2,\dotsc,w_n)$$

where each weight corresponds with an input.


Putting this all together, we can describe the activation function:

$$
f(x) =\begin{cases}
1& \text{if $w \cdot x>0$},\\
0& \text{otherwise}.
\end{cases}
$$

where \\( w \cdot x \\) is the dot product \\( \sum_i w_i \cdot x_i \\).


If we take a network that has more weight with respect to shape than to colour,
say \\( w = (0.5, 2) \\), and ask it to classify a fruit that is close to red,
but line-like, such that the input vector is \\( x = (0.9, -0.8) \\), then we
get the summed input:

$$\sum_i w_ix_i = 0.5*0.9 + 2*-0.8 = 0.45 - 1.6 = -1.15$$

which is less than \\( 0 \\), thus \\( f(0.9, -0.8) = 0 \\).  

According to this network, the fruit is a `banana`.

{% include image.html img="img/single-layer-perceptron/perceptron-f.png" title="Perceptron f" caption="Perceptron f says the red, line-like object is a banana." %}


Another network has a different activation function, \\( g \\), that is
similar to \\( f \\), but its weights are distributed differently. This
network values both inputs almost equally, but thinks colour is slightly
more relevant, say \\( w = (1.1, 1) \\). The input remains the same.
Then it follows that:

$$\sum_i w_ix_i = 0.9*1.1 + 1*-0.8 = 0.99 - 0.8 = 0.19$$

which is greater than \\( 0 \\), thus \\( g(0.9, -0.8) = 1 \\).  

Unlike the previous network, this network says the fruit is an `apple`.

{% include image.html img="img/single-layer-perceptron/perceptron-g.png" title="Perceptron g" caption="Perceptron g says the red, line-like object is an apple." %}

Two networks can classify the same object differently if the distribution of
weights differs.  
Which one of the two is better depends on which can classify more objects
correctly, or has a higher **accuracy**, when used on a larger group of objects.  

Remember that humans are said to be able to perfectly classify fruit as `apple` or
`banana`, meaning they have a 100% accuracy on the task. This is assumed to make
the next stage, learning, easier and more trustworthy, but such high accuracy will
in practice almost never be reached in a single-layer perceptron, or any neural
network for that matter.

Still, any improvement in accuracy is better than the simple guesswork the aliens
rely on now. We can try to change the weights randomly and test the resulting
networks and keep the one with the highest accuracy as our classifier of choice.
More efficient is to progress with a certain plan, which will be a **learning
rule**.

## Learning rule
In order to increase the accuracy of our classifier, we want to change the
weights in the network in such a way that it will classify more objects
correctly. These values are changed based on a learning rule, which is an
algorithm that is repeated until it is stopped manually when a sufficiently
accurate classifier is achieved, or after a certain amount of iterations.

The steps in the algorithm are:

1. Given an input \\( x_j = (I_1,I_2,\dotsc,I_n) \\), the perceptron returns an
output \\( y_j(t) \\) for the \\( j \\)th training input vector.
2. The correct output \\( d_j \\) is given.
3. Compare the outputs \\( y_j \\) and \\( d_j \\), change the weights
\\( w = (w_1,w_2,\dotsc,w_n)\\) only if the outputs are different.

{% include image.html img="img/single-layer-perceptron/training-algorithm.png" title="Training algorithm" caption="A simple training algorithm: calculate some output, compare it to the correct output and update the weights if necessary." %}


The correct output \\( d_j \\) in step 2 is given by the perfect classifiers, the
humans. We know that humans are slow to respond, but instead of having to ask
them for every sample we want to take, we can just ask for an acceptable amount
of correct outputs and then train a perceptron with that. Humans will never have
to be spoken with again after the perceptron can classify on its own.


Corresponding training vectors and correct outputs form a **sample**.

We call \\( D \\) a training set of \\( s \\) samples such that

$$D = {(x_1,d_1),(x_2,d_2),\dotsc,(x_s,d_s)}$$


If the perceptron says an object is an `apple` and the human confirms this, the
perceptron obviously has a good accuracy and no changes are needed. If the
perceptron says it is a `banana`, we need to change the weights to be on our way
to improved accuracy.


The amount by which we need to change the weights depends on a **learning rate**
\\( \alpha \\), where \\( 0 < \alpha \leq 1 \\). The higher the learning rate, the
faster radical changes are made. This will speed up the training, but it will
also result in a periodic oscillation around the optimum when it is reached.

The weight of input \\( i \\) is changed depending on the amount of **error** (the
difference between the correct output and the calculated output):

$$w_i(t+1) = w_i(t) + \alpha(d_j-y_j(t))x_{j,i}$$

where \\( w_i(t) \\) is the weight \\( i \\) at time \\( t \\).


Given our first perceptron \\( f \\) with weights \\( w = (0.5, 2) \\) at
\\( t = 0 \\),  
the training input vectors \\( x = ((0.9, -0.8), (0.8, -0.5)) \\),  
the correct outputs \\( d = (1, 1) \\),  
and the learning rate \\( \alpha = 0.1 \\),  
we can calculate the first iteration \\( t = 1 \\):


First, we calculate the output. As the weights and inputs are similar to our
first example above, we get \\( y_1(1) = 1 \\). Or `apple` (recall that
\\( y_j = 1 \\) means that the object is an `apple`, and \\( y_j = 0 \\) means
that the object is a `banana`).

Looking at our correct outputs, we see that \\( d_1 = 1 \\), thus
the correct class is `apple`.

The outputs are equal, thus no changes have to be made in the weights. This is
seen clearly when filled in the update rule, as \\( d_1-y_1(1) = 0\\).

$$
\begin{equation}
\begin{split}
w_i(1) &= w_i(0) + \alpha(d_1-y_1(1))x_{1,i}\\
&= w_i(0) + \alpha*0*x_{1,i}\\
&= w_i(0) + 0\\
&= w_i(0)
\end{split}
\end{equation}
$$


Now, for a second iteration, we again calculate the output. Note that the weights
are unchanged.

$$\sum_i w_ix_i = 0.5*0.8 + 2*-0.5 = 0.4 - 1 = -0.6$$

which is less than 0, thus \\( y_2(2) = 0 \\). Or `banana`.

Looking at our correct outputs, we see that \\( d_2 = 1 \\), thus
the correct class is `apple`.

The object is classified wrongly and changes in weights must be made.

The first weight (that of the *colour* input) is changed as follows:

$$
\begin{equation}
\begin{split}
w_1(2) &= w_1(1) + \alpha(d_2-y_2(2))x_{2,1}\\
&= w_1(1) + 0.1*(1 - -0.6)*x_{2,1}\\
&= 0.5 + 0.1*1.6*0.8\\
&= 0.5 + 0.128\\
&= 0.628
\end{split}
\end{equation}
$$

Now the second weight (that of the *shape* input) is changed as follows:

$$
\begin{equation}
\begin{split}
w_2(2) &= w_2(1) + \alpha(d_2-y_2(2))x_{2,2}\\
&= w_2(1) + 0.1*(1 - -0.6)*x_{2,2}\\
&= 2 + 0.1*1.6*-0.5\\
&= 2 + -0.8\\
&= 1.92
\end{split}
\end{equation}
$$

thus \\( w = (0.628, 1.92) \\).


The first input, that of colour, has now gotten more weight, while the shape
input dropped in importance.
After a training period of several iterations, you will end up with a certain
distribution of weights \\( w \\) for which the accuracy of the classifier
has most likely improved. This is how a perceptron is trained with a learning
rule.

## Closing remarks

After all this effort, you and your alien friends can watch in peace how humans
devour their fruit, while classifying those with a higher accuracy than mere
guesswork.


There are several issues with perceptrons and there are some remarkable points.
Consider what happens if humans are not able to perfectly classify their fruit.
Then we sometimes get a faulty correct output \\( d_j \\). This will result in
our perceptron training in the wrong direction. It is important that the correct
outputs are as pure as possible.


The aliens have their own kind of fruit that they classify differently.  
They define `bpple` as something that is yellow and circle-like, and they define
`aanana` as something that is red and line-like.  
This results in a classifier that is different from the classifier humans use to
classify `apple` and `banana`.  
The same objects can be classified as different things simultaneously depending
on the class definitions.

{% include image.html img="img/single-layer-perceptron/bpple+aanana.png" title="Bpple and Aanana" caption="The classes Bpple and Aanana are be placed along the colour and shape scale differently from Apple and Banana." %}


A limitation of a perceptron is that not all classification problems are
**linearly separable**. These problems cannot be divided by a simple line, unlike
the apple-banana classification.  
If we add the class `pear` to the problem, a fruit that is yellow-ish and can be
anywhere in between circle-like and line-like, it cannot be split with a simple
line. Say that the problem now is to identify pears. So either an object is a
`pear`, or it is not. We cannot devise a single-layer perceptron that is able to
do the task.  
This limitation is solved with multilayer perceptrons, which will be discussed in
the [following post]({% post_url 2015-08-26-Multilayer-Perceptron %}).

{% include image.html img="img/single-layer-perceptron/add-pear.png" title="Add Pear" caption="Pear or not? A single-layer perceptron cannot help out here." %}

## Further reading

For more information on this topic, consider these helpful links:

A Wikipedia page on perceptrons:
[https://en.wikipedia.org/wiki/Perceptron](https://en.wikipedia.org/wiki/Perceptron)  
A Wikipedia page on activation functions:
[https://en.wikipedia.org/wiki/Activation_function](https://en.wikipedia.org/wiki/Activation_function)  
Extra examples of a single-layer perceptron:
[http://computing.dcu.ie/~humphrys/Notes/Neural/single.neural.html](http://computing.dcu.ie/~humphrys/Notes/Neural/single.neural.html)

[Neural Networks - Part 2: Multilayer Perceptron]({% post_url 2015-08-26-Multilayer-Perceptron %}) is available.
