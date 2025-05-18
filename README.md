# Vegvísir
Customized transformer model for use with my, in development, LLM 'Magi'.

Intended to be trained on different sources of danish literatture, interpersonal interactions and job postings.

To reduce cost of renting datacenter tier GPUs for training, I intend to only deploy such servers when I am confident in the procedure and model,
for that reason I have setup a secondary desktop running ProxMox, which allows me to deploy VMs where I can directly allocate a full GTX 1060 each for accelerated compute 
compared to my laptops. These comparatively powerful and effecient GPUs will allow me to test models with fewer tokens, before training a larger model on enterprise gear.

PyTorch takes advantage of the CUDA cores of the GTX cards, and I expect orders of magnitude faster PoCs.

https://github.com/users/ekka-github/projects/2
Progress on Vegvísir can be found here, in leau of a journal.



When the finished LLM, Magi, is completed - it can be found on my website ekka.dk
Inquieries can be submitted to kontakt@ekka.dk



C. Pedersen.
