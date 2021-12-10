# Gamifier: An Ensemble Program for the Game Design Process
_We present the Gamifier ensemble program, combining 3 deep learning models - ThinkerRNN, PolyGen, and PaletteGAN - to generate 3D meshes and color palettes for relevant objects given one word as the input._

         
<img width="940" alt="Screen Shot 2021-12-10 at 7 15 41 AM" src="https://user-images.githubusercontent.com/75775661/145577205-d2edb198-f7f0-4032-899e-b8d09433945a.png">


# Introduction
The game design process has historically been a tedious one: from modelling thousands of scene contributions (either in 2D or 3D), to creatively producing unique ways to enhance the objects added to a scene. We are living in a world demanding an increasing amount of realism in the games we play. Additionally, with a correlated increase in computation power, many games of today require a higher level of complexity and intricacy within the objects used to build up the imaginative world of the game. 

Hence, we present Gamifier: an ensemble program consisting of three deep learning models designed to simplify the game design process. The first of the models is a RNN language model named “Thinker” — trained on both Wikipedia articles and two Harry Potter books — that takes in a user inputs (say, a single idea for an object to be added to the game) and outputs a list of ideas that are related to the first idea. Once this list has been created, it is fed into DeepMind’s PolyGen model — an “Autoregressive Generative Model of 3D Meshes” — which generates 3D objects as meshes from text descriptions (DeepMind, [PolyGen](https://github.com/awesome-davian/Text2Colors)). Additionally, the list of ideas is inputted into a Conditional Generative Adversarial Network (cGAN) — rebuilt and re-architectured following inspiration from the network “Coloring with Words: Guiding Image Colorization Through Text-based Palette Generation” of Bahng et al. and the corresponding PAT Dataset (Bahng et al., [Coloring with Words](https://github.com/awesome-davian/Text2Colors)) — in order to output a set of color recommendations for the objects outputted by the initial ThinkerRNN model.  Through combining these three models, we nicknamed the new umbrella model “Gamifier”, as we believe it will simplify the game design process and unlock a range of potential applications in 3D scene creation, augmented reality, and virtual reality.


# Notes
This is a final project ([View on Devpost](https://devpost.com/software/gamifiers-deep-3d-mesh-generation-for-game-design)) for CS1470 Deep Learning, Fall 2021, Brown University. 
