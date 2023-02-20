# Adversarial attacks in image recognition

## Adversarial attacks  
1. Non-targeted vs. targeted attacks: Non-targeted attacks aim to cause a model to misclassify an image, while targeted attacks aim to cause a model to classify an image as a specific target class. 

2. Gradient-based attacks: Gradient-based attacks involve computing the gradient of the loss function with respect to the input image, and then modifying the image in the direction of the gradient to cause the model to misclassify the image. Examples: Fast Gradient Sign Method, Basic Iterative Method.

3. Evolutionary algorithms: Evolutionary algorithms involve iteratively generating and evolving images to create adversarial examples that cause a model to misclassify the image. Example: Genetic Algorithm.

4. Transferability attacks: Transferability attacks involve creating adversarial examples on one model and then showing those examples to a different model to cause it to misclassify the image. Example: Single Pixel Attack.

5. Defense-generating attacks: Defense-generating attacks involve generating adversarial examples to improve a model's robustness to adversarial attacks. Examples: Virtual Adversarial Training.

6. Physical/object-specific attacks: Physical attacks involve modifying the input data in a way that is imperceptible to humans, but can cause the model to misclassify the data in the real world. Object-specific attacks involve designing an attack specifically for a particular object. Example: adversarial patches (printing out a patch and physically placing it in the real world).

resources:
- "Adversarial Machine Learning" by Battista Biggio and Fabio Roli
- "Explaining and Harnessing Adversarial Examples" by Ian Goodfellow, Jonathon Shlens, and Christian Szegedy
- "Towards Deep Learning Models Resistant to Adversarial Attacks" by Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, and Xiaolin Hu
gradient-based and non-gradient-based attacks

## Image recognition algorithms  
YOLO  


## Packages  
https://github.com/jaypmorgan/Adversarial.jl 


## Code examples
https://github.com/ilkerkesen/Taarruz.jl  
https://github.com/gitter-badger/FaceCracker.jl  




## Useful links  
https://arxiv.org/pdf/1905.08614.pdf  
https://arxiv.org/pdf/1907.10456.pdf  
https://arxiv.org/abs/1412.6572
https://arxiv.org/abs/1909.08072  



## Real-life cases that seem interesting and somehow connected
https://www.nytimes.com/2019/04/14/technology/china-surveillance-artificial-intelligence-racial-profiling.html  
https://www.biometricupdate.com/202302/designers-take-on-facial-recognition-with-adversarial-fashion
https://uk.news.yahoo.com/400-sweater-tricks-facial-recognition-080000503.html?guccounter=1&guce_referrer=aHR0cHM6Ly9kdWNrZHVja2dvLmNvbS8&guce_referrer_sig=AQAAAD0-JAbxgtRG-4DftJhvbSQd970FNuOiASbGOnqURQ-hzEdX1pOxUUYLpcDT1mo7X1lJHX-imwcGDJ1RjEKaTbTCDrOe62Y_Ql6PREMwsMO-okG4rLMc1Qs1XWR4pxEXR08wZuatw7MqqouaMpXCPN0u6F4_BfGb2tsidQR8IIGU
https://www.biometricupdate.com/202207/new-adversarial-mask-designs-evade-facial-recognition-systems
https://www.biometricupdate.com/202302/indian-officials-use-facial-recognition-to-identify-thousands-of-fraudulent-sim-cards
https://nypost.com/2023/02/12/retailers-busting-thieves-with-facial-recognition-tech-used-at-msg/
https://news.bloomberglaw.com/us-law-week/how-ai-and-facial-recognition-can-chill-access-to-justice
https://futurism.com/the-byte/casinos-facial-recognition-gambling
https://www.foxnews.com/tech/how-stop-facial-recognition-cameras-monitoring-your-every-move


For instance, recent research data from Israeli researchers showed that fabric face masks covering the nose and mouth and printed with adversarial patterns evaded facial recognition systems more than 96 percent of the time.

At the same time, companies and governments are actively working to improve and anti-spoofing abilities of facial recognition systems. A camera recently developed by Sony, for example, is designed to protect against image manipulation and spoof techniques like face morphing.

