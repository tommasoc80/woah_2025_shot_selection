# Learning from Disagreement: Entropy-Guided Few-Shot Selection for Toxic Language Detection

This repository contains the code and links to the datasets used for the paper "Learning from Disagreement: Entropy-Guided Few-Shot Selection for Toxic Language Detection" accepted at WOAH 2025mco co-located with ACL 2025.

The datasets we used can be retrived at their respective repositories: 

- EDOS: https://github.com/rewire-online/edos
- Brexit (part of LeWiDi 2023 - original name HS-Brexit_dataset): https://github.com/Le-Wi-Di/le-wi-di.github.io/blob/main/LeWiDi-2ndEDITION-data_post-competition.zip
- GAB Hate speech corpus: https://osf.io/edua3/
- Multi-Domain Agreeement Dataset (part of LeWiDi 2023 - original name MD-Agreement_dataset): https://github.com/Le-Wi-Di/le-wi-di.github.io/blob/main/LeWiDi-2ndEDITION-data_post-competition.zip
- SBIC: https://maartensap.com/social-bias-frames/

The /data/ folder contains the followings:
- the format we have used to compute the entropy scores with [MACE](https://github.com/dirkhovy/MACE)
- the plots in Appendix A
- the full output from [MACE](https://github.com/dirkhovy/MACE)
- the code to extract the shots for the ICL experiments
  
Additionally, we make available the following data distributions *only* for replicability purposes for the GAB and SBIC datasets. For GAB, we make available the splits for train and dev we have used for the experiments with [HateBERT](https://huggingface.co/GroNLP/hateBERT). For SBIC, we make avaialble the data with the aggrgated labels for the offensive lagauge detection task we report in all experimnts (with LLMs and with HateBERT) in the paper.

All predictions of all the models are avaialble upon request (mail to t [dot] caselli [at] rug [dot] nl).
