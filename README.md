# KDT-CE
Source codes of the article:  

S. Li, P. Dong and R. Li, "Knowledge Distillation Transformer for XL-MIMO Channel Estimation," IEEE Internet of Things Journal, vol. 13, no. 12, pp. 26652-26665, 15 June15, 2026.

Please cite this paper when using the codes.

# Instructions

Reading the below sections in the written order will help better understand all the codes.

 ## Python

**ChannelData_generation.m**
Gemerate the training dataset and the test dataset.

**student.py**
Student.py uses knowledge distillation to enable the lightweight student model to learn from the pre-trained teacher model.

**512student.py**
512student.py contains the training code for the student model under the condition of 512 antennas.

**ADCtest.py**
ADCtest.py is used to perform ADC tests on the trained student network.

**SSL.py**
SSL.py represents the self-supervised learning baseline method proposed in the paper.



# Environment
These models are implemented in Keras, and the environment setting is:

-   Python 3.7.1
-   TensorFlow-gpu 2.3.0
