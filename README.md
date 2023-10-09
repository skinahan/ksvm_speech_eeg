# EEG Classification Reveals Impaired Speech Motor Planning in Stuttering Adults

Purpose: This study explores speech motor planning in adults who stutter (AWS) and adults who do not stutter (ANS) by applying machine learning algorithms to electroencephalographic (EEG) signals. In this study, we developed a technique to holistically examine neural activity differences in speaking and silent reading conditions across the entire cortical surface. This approach allows us to test the hypothesis that AWS will exhibit lower separability of the speech motor planning condition. Method: We used the silent reading condition as a control condition to isolate speech motor planning activity. We classified EEG signals from AWS and ANS individuals into speaking and silent reading categories using kernel support vector machines. We used relative complexities of the learned classifiers to compare speech motor planning discernibility for both classes. Results: AWS group classifiers require a more complex decision boundary to separate speech motor planning and silent reading classes. Conclusion: These findings indicate that the EEG signals associated with speech motor planning are less discernible in AWS, which may result from altered neuronal dynamics in AWS. Our results support the hypothesis that AWS exhibit lower inherent separability of the silent reading and speech motor planning conditions. Further investigation may identify and compare the features leveraged for speech motor classification in AWS and ANS. These observations may have clinical value for developing novel speech therapies or assistive devices for AWS.

[Validation Accuracy Figure](https://github.com/skinahan/ksvm_speech_eeg/blob/main/results/figures/42_Validation%20Accuracy%20as%20F(C)%20with%20Gamma%200.005_3.png)

# ksvm_speech_eeg

This repository applies Kernel SVM to classify EEG trials into Speaking and Silent Reading conditins.

Note: 
The EEG dataset used for this project was described in [Daliri & Max, 2018](https://doi.org/10.1016/j.cortex.2017.10.019). To obtain access to this data, please contact the authors directly.

# How to use (Windows)

1. Clone this repository
2. Install Anaconda
3. Create and activate a conda environment using the provided requirements.txt
   - conda create --name your_env_name --file requirements.txt
   - conda activate your_env_name
4. Run main.py using python

- Analysis and plotting methods can be found in the data_analysis module. 
- Experimental results (not raw data) can be found in the renamed_pkls directory.
   - These files can be loaded and examined using the pandas package.

# Questions

Contact: skinahan {at} asu {dot} edu
