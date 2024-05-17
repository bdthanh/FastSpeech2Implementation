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

The pretrained acoustic model and its corresponding dictionary are also neccessary. Run the command: 
```
mfa model download acoustic english_mfa
mfa model download dictionary english_us_mfa
```

Then, run the following command to align the dataset:
```
mfa align data\pre_mfa_LJSpeech english_us_mfa english_mfa data\post_mfa_LJSpeech
mfa train data\pre_mfa_LJSpeech\LJSpeech lexicon\librispeech-lexicon.txt data\post_mfa_LJSpeech\new_mfa.zip --output_diretory data\post_mfa_LJSpeech
```

After that, run the processing scripts to get duration, pitch and energy targets:
```
python src\post_mfa_process.py --config [PATH TO config.yaml]
```