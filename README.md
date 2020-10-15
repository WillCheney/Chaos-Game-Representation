# Chaos Game Representation
Chaos game representation(CGR) is algorithmn which can be used to visualize Markov chains of a system.

## Chaos Game shape
When the chaos game is applied to system with equal probability outcomes it can generate fractals such as the Sierpiński triangle.<br>
Chaos Game shape.py generates a chaos game representation image of equal probability n-outcome Markov chain.<br>

Examples:<br>
![examples](https://github.com/WillCheney/Chaos-Game-Representation/blob/master/Chaos%20Game%20example-01.png)


## Chaos Game Representation of DNA Sequences
Chaos game representation can also be applied to genetic sequences to create a "fingerprint" of a genome's intrinsic sequence probabilties.<br>
DNA_CGR_generation.py contains functions to generate a chaos game representation of a DNA sequence and functions to randomly sample a genome and build dataset of CGR images.<br>

Example: 50kb CGR image of Human genome, Pattern shows absence of sequential C-G base pairs to prevent CpG methylation.<br>


## Using Machine Learning to Classify Genomes.
Since organism have evolved, either by drift or selection, unique base pair usage probabilties different organisms should produce unique CGR of their genome.<br>
We can use differences and image processing for genome classification.<br>
Genome Classification using CGR.ipynb is an interactive python notebook which can effectively(~95%) classify Human/Yeast genomes or genetically similar Saccharomyces/Candida genomes.<br>
It also contains the steps to produce a CGR dataset and train a neural network on that dataset.<br>
CGR_image_classifier.py contains functions to train a neural network on CGR image data and functions to classify unknown CGR genome images.<br>


Example: Classification of unknown CGR genome using neural netowrk
![examples](https://github.com/WillCheney/Chaos-Game-Representation/blob/master/Neural%20netowrk.png)
