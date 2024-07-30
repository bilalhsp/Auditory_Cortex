
### Table of Contents
2. [Pretraining on Natural Sound Dataset](#2-pretraining-on-natural-sound-dataset)  
    2.1. [Data Preparation](#21-data-preparation)  
    2.2. [Model Pretraining](#22-model-pretraining)
3. Something else



### Summary of saved results:
Need to make a table of this!!!
#### STRF:
 - actual baseline using wavelet, using optimal lags fitted  by cross-validated.
    timit
    mVocs
 - Baseline using mel-spectrogram, using optimal lags fitted by cross-validated.
    mel_spectrogram
    mVocs_mel_spectrogram

#### Group of seven networks:
 - LPF_features_20: Contains correlation results for predicting spikes at 10 ms,
                    while features are low pass filtered at different bin widths
                    (downsampled-upsampled). Bin width column carries a different
                    meaning here, it corresponds to the bin width the features
                    were downsampled to before being upsampled back to 10 ms.
                    Bin widths = [20, 30, 40, 50, 60, 80, 100, 200,
                                300, 400, 500, 600, 700, 800]

#### 06-27-24: Constant-Q transform (wavelet transform) vs mel-spectrogram: 
- naplib python package provides an implementation of spectrogram that they refer to as constant-Q transform, (I think it is a wavelet transform, I'll go thru the paper that they refer). They claim that it is more like human auditory processing. Originally this was used for fitting the STRF baseline. 
- I fit STRF baseline using the mel-spectrogram, just to compare it with naplib's version of spectrogram. And mel-spectrogram performs poorly compared to the original one. Median correlation drops to 0.38 by using mel-spectrogram provided by speech2text processor. (Median correlation was original 0.53)
- To test the difference of spectrogram, I used naplib spectrogram as input to wav2letter_spect (untrained version) and computed correlations using both timit and mVocs stimuli.
- Since networks trained on speech datasets as well as wav2vec2 trained on Audioset (which we believe is dominated by speech) performed worse than spectrogram (wavelet) baseline, we need to a network that is trained on natural sounds. We did find network pretrained on the entire Audioset but we want to take out speech (and may be music as well) so that network is trained to process natural sounds other than speech. We'll use that network to extract representations of monkey vocalizations. But it is going to take time to train a network from scratch, so I proposed we first try a network pretrained on music (available on Hugginface, music2vec). But we should find out is music processing going to help us? How is monkey vocalization different from human speech? Is it more like music? 
- I trained STRF for timit and mVocs using both wavelet and mel-spectrogram. 
#### wav2letter_spect (untrained): 
- spectrogram: trained on timit, using wavelet transform (CQT)
- mVocs_wavelet: trained on mVocs, using wavelet transform (CQT)

### 07-01-24:
#### Huggingface Audio processing:
https://huggingface.co/welcome Create a new repo and make it available for others to use. 

### 07-09-24:
#### LPF_features_10: 
Predicting neural spikes at 10ms, using features low pass filtered at frequencies corresponding to bin_width specified in the file. (Note: use of bin width specifying low pass filtering the features is different than earlier use of predicting at bin_width. Here we are  predicting at fixed bin width =10ms mentioned in identifier).
 

Correlation peaks for models:
<div align="center">

| model names | peak bin width (sampling rate) |
| :--------    |:---: |
| whisper_tiny | 20 ms|
| whisper_base | 20 ms|
| deepspeech2 | 20 ms|
| speech2text | 20 ms|
| wav2letter_modified   |   30 ms| 
| wav2vec2 | 20 ms|
| wav2vec2_audioset | 20 ms|

</div>

### 2. Pretraining on Natural Sound Dataset
#### 2.1 Data Preparation

**07-18-24:**  
I have been working to train a wav2vec2 network on audioset (speech and music removed). I started by using [audioset huggingface](https://huggingface.co/datasets/agkphysics/AudioSet) but that dataset was huge. I had to download 2.2TB of data on /scratch. I was able to speed up download by using multiprocessing supported by huggingface datasets. But filtering that dataset, to get rid of speech and music sounds, was even bigger challenge as filtering function did not support multiprocessing and I wasn't able to do filtering and saving back to disk within 4 hours, wall time provided by *stanby* queue of Gilbreth (although I figured later that appying *filter* method naively loads all the features of the dataset, that includes reading/loading audio array from the disk, which makes it extremly slow. Instead of doing that if we are to filter based on labels only then I should read only the labels using **dataset['human_labels']**, and apply filtering logic on that column only and using the list of booleans we can use **dataset.select([good_ids])** to get a subject (filtered) of dataset. This ran within a few minutes as compared to couple of hours). Even then for space contrained settings or while looking to use a very small subject of audioset, it doesn't make sense to download more than 2TB of data. 

- Downloaded audioset from [huggingface](https://huggingface.co/datasets/agkphysics/AudioSet)  
- Tried to move dataset to *phocion* to filter out speech and music.
- Could only copy 30% of 2.2TB in 30 hours.
- I wrote my own implementation to take the metadata and filter out the unwanted labels, download audios directly form youtube links. 
- My repo [audioset_utils](https://github.com/bilalhsp/audioset_utils) that provides functionlity of filtering, downloading and creating huggingface dataset.
- I saved everything as *wav* files, sampling at **48000 Hz**. 
- **Excluded labels**: 
    ```python
    [
        # speech related labels...
        'speech', 'speak', 'conversation',
        'monologue', 'shout', 'yell',
        'singing', 'chant', 'television',
        'radio',
        # music instruments...
        'music', 'drum', 'opera', 'jazz',
        'disco', 'reggae', 'country', 
        'saxophone', 'flute', 'instrument',
        'trumpet', 'piano', 'sitar', 'guitar',
    ]
- There were *334,597* examples after excluding above labels, each example is 10s long clip, so it should be *929* hours of audio. (It turned out that some examples were shorter than 10 seconds, so I had to filter out very short examples i.e. less than 5 seconds long)
- But we lose some examples due to link unavailble anymore, made private or other reasons, so ended up downloading *282790* examples, comprising *786* hours of audio.
- Filtering out short (less than 5 sec long) examples, we are left with dataset as below:

<div align="center">

| split    | # of examples | duration (hours) |
| :---: | :---: | :---: |
| training | 281099 | 780 |
|test      | 8437 | 23 |

</div>

#### 2.2 Model Pretraining

**07-18-24:**  

<!-- <div align="center">

| Header 1   | Header 2   | Header 3   |
|------------|------------|------------|
| Row 1 Col 1| Row 1 Col 2| Row 1 Col 3|
| Row 2 Col 1| Row 2 Col 2| Row 2 Col 3|

</div> -->
