## Preprocessing

First, we need to do some initial processing for later alignment process, please run
```
python src\pre_mfa_process.py --config [PATH TO config.yaml]
```
 
According to the FastSpeech 2 paper, the authors used [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) to align the utterances and the phoneme sequences. First, we need to download the MFA library:
```
conda config --add channels conda-forge
conda install montreal-forced-aligner
```

Run the following command to train the new MFA acoustic model using librispeech-lexicon.txt, which takes about 3-5 hours:
```
mfa train data\pre_mfa_LJSpeech\LJSpeech lexicon\librispeech-lexicon.txt data\post_mfa_LJSpeech\new_mfa.zip 
```
Then, run the following command to align the dataset, and this will take about another 4-5 hours:
```
mfa align data\pre_mfa_LJSpeech\LJSpeech data\post_mfa_LJSpeech\librispeech-lexicon.dict data\post_mfa_LJSpeech\new_mfa.zip data\post_mfa_LJSpeech --clean
```


After that, run the processing scripts to get duration, pitch and energy targets:
```
python src\post_mfa_process.py --config [PATH TO config.yaml]
```