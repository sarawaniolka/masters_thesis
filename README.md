# masters_thesis

# language = Julia
# method = Neural Network

## Idea 1:  
Generating a short sample of a song based on provided genre.  
1. Data: MIDI, MP3 or music scores files of songs from given genres. Maybe they should be converted into spectograms? Possible sources:  
    a) Free Music Archive  
    b) MuseScore  
    c) Royalty Free Music
MP3 files can be converted to MIDI
2. Architecture: 
    a) Generative Adversarial Network (GAN)
    b) Variational Autoencoder (VAE)
    c) Recurrent Neural Netword (RNN) with a long short-term memory
3. Starting point: user provides the genre but what else? First notes, random vector, a song?
4. Post-processing of the sound.

Problems: large amount of data, a lot of computational resources

## Idea 2 (similar, easier)
Classifying a given song to a genre. This is a classification problem instead of a generation problem, so it sounds easier to me. However, it needs similar data for training.
1. Data: same as before
2. Architecture: learning the features of each genre.
    a) Multi-layer Perceptron (MLP)
    b) Convolutional Neural Network (CNN)
    c) Recurrent Neural Network (RNN)
3. Testing
4. Deploying: if it does well maybe I could add the possibility to clasify new songs, as they get added to spotify, youtube music or other platforms?

Problems: large amount of data, some songs belong to multiple genres, so not sure how to address it

## Idea 3
Music style / artist transfer. I recently saw a video about it, looked interesting but I am not sure how hard it would be to actually do it. So, I basically saw that someone created an algorithm which was capable of transfering one singer's voice into a song sung by a different artist, e.g. Michael Jackson's Thriller was sung by Lady Gaga and it sounded quite realistic.
I think it's called a voice transfer. An example is here:
https://github.com/andabi/deep-voice-conversion
Two neural networks would be required.
Sounds cool to me but I am not sure if it is doable.
Other useful sources:
https://arxiv.org/pdf/1904.06590.pdf
https://towardsdatascience.com/voice-translation-and-audio-style-transfer-with-gans-b63d58f61854


