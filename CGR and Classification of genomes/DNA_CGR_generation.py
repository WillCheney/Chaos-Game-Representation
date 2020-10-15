import math
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO


def chromosome_load(filename):
    #Creates list of chromoesome sequences from fasta file
    #Genomes can be downloaded at NCBI
    
    #Parameters:
    #   filename: path of .fna genome file
    #
    #Returns:
    #   chromosomes: list of chromosomes from genome file where each element is the chromosomes sequnce

    with open(filename) as genome_file:
        chromosomes = []
        for record in SeqIO.parse(genome_file, "fasta"):
            if record.id[0:2] == 'NC': #NC is chromosome annotation
                chromosomes.append(record)
    return chromosomes



# Generates a random sample of genome sequences of size bps
# Bps determines length of sample (number of base pairs)
# Chr_sample determines how many chromosomes to sample
# sequence_sample determines how many samples per chromosomes
# total number of samples = chr_sample * sequence_sample
# returns list of samples with each index contain a continous DNA seqence of size bps

def seq_sample(chromosomes, bps = 50000, n = 30):
    #Parameters:
    #   chromosomes: list of DNA sequences to sample from
    #   bps: desired length of continous sample
    #   n: number of samples
    #
    #Returns:
    #   samples: list of random continous DNA samples from chromosomes

    
    samples = []
    
    for k in range(n):
        i = np.random.randint(0, len( chromosomes))
        j = np.random.randint(0, len( chromosomes[i]))
        if (j + bps) > len( chromosomes[i]): #Check for out of bounds error  
            j = len( chromosomes[i]) - bps
        samples.append( chromosomes[i].seq[j: (j + bps)])
    
    return samples
        


# In[52]:

# Generates points for CGR representation of sequence
# Steps through Chaos game for DNA sequence

def dna_cgr(sequence):
    #Parameters:
    #   sequence:  DNA sequence to iterate chaos game over
    #
    #Returns:
    #   x: list of x coordinates from each step
    #   y: list of y coordinates from each step
    
    sequence = sequence.lower()
    
    #define vertex coordinates
    C = (-1,1)
    G = (1,1)
    A = (-1,-1)
    T = (1,-1)
    #inital point
    x = [0]
    y = [0]

    for i in list(sequence):
        #for each step add a new point half way between the previous point and the vertex for bp i
        if i == "n":
            continue
        elif i == "a":
            nextx = (x[-1] + A[0])/2
            nexty = (y[-1] + A[1])/2
            x.append(nextx)
            y.append(nexty)

        elif i == "t":
            nextx = (x[-1] + T[0]) / 2
            nexty = (y[-1] + T[1]) / 2
            x.append(nextx)
            y.append(nexty)

        elif i == "c":
            nextx = (x[-1] + C[0]) / 2
            nexty = (y[-1] + C[1]) / 2
            x.append(nextx)
            y.append(nexty)

        elif i == "g":
            nextx = (x[-1] + G[0]) / 2
            nexty = (y[-1] + G[1]) / 2
            x.append(nextx)
            y.append(nexty)
            

    # returns list of points
    return x,y


# In[122]:

#Shows CGR image
def show_cgr(x):
    #Parameters:
    #   x: DNA sequence
    
    x,y = dna_cgr(x)
    plt.scatter(x,y, s = 0.2, color = 'black', alpha = 1)
    plt.axis('off')
    plt.show()
    plt.close()
    



#Loop to generate data set of CGR images
def generate_images(samples, title):
    #Parameters:
    #   samples: list of DNA sequences from which to generate Chaos game representations
    #   title: title of saved image file will increment numerically 
    #


    
    counter = 1
    for i in samples:

        x,y = dna_cgr(i)
        plt.scatter(x,y, s = 0.1, color = 'black', alpha = 1)

        if len(samples) == 1:
            title = title
        else:
            title = title + str(counter)
        plt.axis('off')
        fig = plt.gcf()
        fig.savefig(title, dpi = 300, bbox_inches='tight')
        plt.close()
        counter += 1

