# Automatic decryption of classical ciphers with neural networks

*This respository is a copy of the private git repository [spring-epfl/spring21-JodokVieli](https://github.com/spring-epfl/spring21-JodokVieli). The code is published with the approval of the [SPRING](https://www.epfl.ch/labs/spring/) lab. However, all data must remain private and is not included in this repo. For demonstration purposes we provide data from English texts from Alice's Adventures in Wonderland and The Bible.*

-----

This repository contains all code related to the semester project *"Automatic decryption of classical ciphers with neural networks"* written by Jodok Vieli under the supervision of Kasra Edalat Nejad in the [SPRING](https://www.epfl.ch/labs/spring/) lab.
 
## Task description

The goal of this project was to build a model `h` that predicts the correct plaintext `plain` given a ciphertext `cipher`. We tackle this challenge by first training a classifier `clf` learning to distinguish between `correct` vs `incorrect` decryption. This `clf` is then used to evaluate decryption attempts. For now, we only consider messages encrypted under the caesar and substitution cipher. The model can easily be extended to learn the `cipher` to `plain` mapping for other weak ciphers.

## Data

For generation of training data we consider messages with from different sources with varying level of complexity:

* Messages from English texts, such as the bible (`bible`) or alice's adventures in wonderland (`alice`)
* ACARS messages (`acars`)

Messages in this context are text samples parsed from the respective text source under a specific alphabet. These messages represent the "negative" class `correct`. The folder `data/` includes pickled `pandas.DataFrame` objects representing train and test sets for `bible` and `alice` data.

A cleaned dataset of ~200,000 messages collected from the ACARS data link network was made available. We differentiate between plain_manual, i.e., a message sent through a manual-user interface by a crew member or plain_sensor, i.e., a message sent by an aircraft's internal sensor systems. Due to data protection agreements the full dataset can not be made available.

We provide a jupyter notebook `data_cli.ipynb` that demonstrates how to generate additional collections of messages from `.txt` files.

### Generating labelled training examples

The previously collected data only contains samples of the `correct` class. Examples of the class `incorrect` need to be generated. A [preliminary survey](https://content.sciendo.com/view/journals/popets/2018/3/article-p105.xml) [1] found that most proprietary encryption approaches rely on weak substitution ciphers. The file `src/ciphers.py` implements examples of weak ciphers such as substitution ciphers and helper functions for key management. We generate examples of `incorrect` by encrypting a messages with a random key and decrypting it with another random key. The following ciphers were used to generate examples of the class `incorrect`:

**Caesar**: The Caesar cipher, also referred to as shift cipher, is a simple substitution cipher in which each letter in the plaintext is replaced by a letter some fixed number of positions down the alphabet, i.e., each letter is "shifted" by a fixed number of positions.

**Substition**: The simple substitution cipher differs from the Caesar cipher in that the cipher alphabet is not simply the alphabet shifted. Each plaintext character is substituted with a random character from the plaintext alphabet.

All ciphers are based on their [secretpy](https://pypi.org/project/secretpy/) implementation. For more information, and a nice introduction to cryptography, see Simon Sing's [The Code Book](https://simonsingh.net/books/the-code-book/) [2].


### Message encoding

The file `src/data_utils.py` contains helper functions to manipulate the message data described above. I particular, functions to load examples of the `correct` class (`load_msgs`), to one-hot encode a list of messages into a binary array (`encode_msgs`) and to generate a `pytorch` compatible `DataLoader` object containing a full dataset with samples of `correct` and `incorrect` based on the specified data id, ciphers and encryption parameters (`gen_dl`).

### Partial decryption

We introduce the notion of partial decryption to get better performance on cracking the substitution cipher. We encode partial decryptions by masking characters which are not decrypted yet with a special character. A `correct` partial decryption is defined as a decryption where all unmasked characters match the original plaintext of the message. The goal of `clf` can be rephrased as distinguishing between `correct` *partial* decryptions and `incorrect` *partial* decryptions. Consequently, we generate datasets of messages partially decrypted up to a specific length.

### Low distance decryption

We further introduce the notion of low distance decryption referring to samples of the class `incorrect` generated to be artificially close to the correct decryption. We expect the addition of such low distance decryption to the dataset to improve the quality of our `clf`, as it gives the classifiers more examples to train where prediction is the most difficult.

## Model

The model `CharCNN` in `src/models.py` provides a `pytorch` implementation of a [character-level convolutional neural network](https://arxiv.org/pdf/1509.01626.pdf) presented by Zhang et al. [3].
The model expects as input an array of one-hot encoded characters and outputs the estimated (linear) class probabilities. 

The Trainer in the file `trainer.py` provides all the training functionality to train (`fit`) and test (`score`) the network and to use it to make new predictions (`predict`).

The network is trained through back-propagation using `adam` optimization on mini-batches of configurable size. Other optimisers, such as the `sgd provided through the pytorch library, can be used but the default training configuration provided the best results and was used across all experiments.

We provide the jupyter notebooks `train_caesar_cli.ipynb` and `train_subst_cli.ipynb` to demonstrate how to train models for the respective ciphers. 

We pretrained various models on different datasets. We provide pretrained models for the following data:

* Six classifiers trained on messages encrypted under the caesar cipher. The classifiers differ in the proportion of the class `incorrect` and the data source they are trained on. The proportion of the class `incorrect` is equal to 1, 5, and 10 times the size of the class `correct` respectively. The data source is either `bible` or `acars`.
* Two sets of classifiers trained on messages encrypted under the substitution cipher. The substitution cipher is only trained on data extracted from `bible`. For each set we train 13 classifiers for partial decryptions with length [2...26]. The first set is trained on regular data. The data for the second set is expanded by additional low distance decryptions as described in the report.

The parameters used to train a classifier are specified in file name. We provide an example to visualize the naming scheme:
The classifier `bible_basiclower_subst_partial10_prop1_lowdist.cnn` was trained using messages extracted from `bible`, parsed with the `basic` English alphabet with `lower` case characters only, encrypted with `substition` cipher, using `partial` decryption of `10` characters with additional `low distance` decryptions added to the class `incorrect`.

The models can be loaded with the helper function `load_model` in the file `model_utils.py`. The use of the models is best visualized in the notebooks `break_caesar_cli.ipynb`and `break_subst_cli.ipynb`.

Unfortunately, the pretrained models cannot be provided on github due to their large size.

## Cracker

The file `src/cracker.py` implements functionality related to recover the correct key given `clf` and `cipher`, such as a key search algorithm (`key_search`) for each cipher. An instantiation of `cracker` uses a `clf` to evaluate decryption attempts. For the substituiton cipher `cracker` recovers the key iteratively for partial decryptions of increasing lengths until the partial decryption corresponds to a full decryption. For each iteration step a different `clf` is used which is trained on partial decryptions of the specific length of that iteration.

We provide the jupyter notebooks `break_caesar_cli.ipynb`and `break_subst_cli.ipynb` to demonstrate how to recover keys for the respective cipher.

## Evaluation

The file `evaluation_utils.py` implements functionality related to evaluating the performance of the different variations of model `h`. In particular we provide functions to compute the accuracy of the classifier `clf` over different message lengths (`eval_model_across_bins`) and to compute the accuracy of the model `h` over different message lengths (`eval_cracker_across_bins`).

We provide the jupyter notebooks `eval_caesar_cli.ipynb` and `eval_subst_cli.ipynb` to demonstrate how to use these functions and to demonstrate the performance of key recovery.

## Future Work

### Insights and pain points

When working on this project we experienced numerous challenges which slowed down our progress. Some of these challengens we want to present here, to help adress these challenges efficiently if they might come up in any future work.

* Sensor generated data from `acars` will always contain a similar sequence of characters right at the start of the message. This leads to unnaturally high performance of the classifier, when assigning the label `correct` or `incorrect`. A cleaned version of this data stripped from the repeating characters can be loaded with `gen_dl` and `load_msgs` by setting `data_id` to `acars-sensor-train` or `acars-sensor-test` respectively.
* The training of classifiers always converges but does not always end up in a reliable classifier. In the notebook `train_subst_cli.ipynb` and `train_caesar_cli.ipynb` we provide plots of the confusion matrix for manual inspection of each classifier.
* The accuracy of a classifier does not always match the performance of the resulting key recovery algorithm. This behaviour is visualized well in the plots comparing accuracies of different classifiers versus accuracies of key recovery algorithms using the same classifiers in the notebook `eval_caesar_cli.ipynb`.
* A classifier might assign a lot of decryptions a high `correct` score. We experienced this behaviour most pronounced when training classifiers to recover the correct key from messages encrypted under the substition cipher. This behaviour severly complicates the automated recovery of the correct decryption. We have not yet been able to overcome this challenge, but carefull generation of additional data such as increasing the proportion of `incorrect` or adding low distance descriptions have proven to reduce this effect to some extent
* Increasing the proportion of `incorrect` makes the training procedure vulnerable to end up with a classifier which alway predicts the majority class. Using class weights when defining the loss function reduces this effect to some extend. However, to find the right balance between proportion and class weights becomes increasingly more difficult with increasing class imbalance.
* Frequent key rotation does not have a large effect on the performance of a single classifier. However, it makes the classifiers perform better when recovering the correct decryption. We usually rotated the key after every message.

### Current status and future directions

Unfortunately the scope of a semesterproject was not enough to fully explore the capabilities of neural networks to automatically decrypt messages encrypted under weak ciphers. Work on this project is halted in the following state:

* Caesar cipher can be automatically decrypted for low and high complexity data.
* Substitution cipher cannot be automatically decrypted yet.
    * We use iterations over different classifiers to decrypt messages.
    * We use the classifiers confidence or the classifiers loss to evaluate decryption attempts. Both perform equally well.
    * The model succeeds to predict the correct partial decryption until partial decryption of length 10. The classifier predicting partial decryptions of length 12 performs significantly worse than other classifiers. This is visualized well in the notebook `eval_subst_cli.ipynb`.

In the following we present next steps to be taken and promissing paths to be explored in any future work:

* Improve reliability of classifiers for specific iterations of breaking the substitution cipher (Iteration of partial decryption length 12 perfoms exceptionally bad).
* Refine generation and proportion of low distance decryptions.
* Find performance metric that better reflects ability of classifier to recover correct decryption than the classification accuracy.
* Automate the evaluation process of trained classifiers. Classifiers performing poorly should automatically be retrained.
* In the current approach we use a special character to mask characters which were not decrypted when generating partial decryptions. The uses of a shadow alphabet instead of a single masking character would increase the capacity of the classifiers.
* An alternative approach would be to train a model to directly predict the correct key.
* Expand approach to decrypt messages encryped under other classical ciphers such as `vigenere` and `columnar`.

## Dependencies

All code is written in `python3`.

The full module list of requirements to setup a virtual environment can be found in `requirements.txt` but the main (non-standard) dependencies of this project are

* torch: A python framework for scalable deep learning
* secretpy: A python library that contains a suite of classical cipher algorithms 

## References

[1] M. Smith, D. Moser, M. Strohmeier, V. Lenders, and I. Martinovic, “Undermining privacy in the aircraft communications addressing and reporting system (ACARS),” Proceedings on Privacy Enhancing Technologies, vol. 2018, no. 3, pp. 105–122, 2018.

[2] Singh, Simon. The code book. Vol. 7. New York: Doubleday, 1999.

[3] X.Zhang, J.Zhao, and Y.LeCun, “Character-level convolutional networks for text classification,” in Advances in neural information processing systems, 2015, pp. 649–657.
