## Preprocessing

First, we need to do some initial processing for later alignment process, please run
```
python ...
```
 
According to the FastSpeech 2 paper, the authors used [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) to align the utterances and the phoneme sequences. First, we need to download the MFA library:
```
conda config --add channels conda-forge
conda install montreal-forced-aligner
```
Then, run the following command to align the dataset:
```
mfa align ... 
```
After that, run the processing scripts to get duration, pitch and energy targets:
```
python ...
```