# "Why the face?": Exploring Robot Error Detection Using Instrumented Bystander Reactions

Maria Teresa Parreira [1,2], Ruidong Zhang [1], Sukruth Gowdru Lingaraju [2], Alexandra Bremers [2], Xuanyu Fang [2], Adolfo Ramirez-Aristizabal [3], Manaswi Saha [3], Michael Kuniavsky [3], Cheng Zhang[1], Wendy Ju [1,2]

1.Cornell University, Ithaca, NY, USA
2.Cornell Tech, New York, NY, USA
3.Accenture Labs, USA

[IMAGE]

How do humans recognize and rectify social missteps? We achieve social competence by looking around at our peers, decoding subtle cues from bystanders — a raised eyebrow, a laugh — to evaluate the environment and our actions. Robots, however, struggle to perceive and make use of these nuanced reactions.  By employing a novel neck-mounted device that records facial expressions from the chin region, we explore the potential of previously untapped data to capture and interpret human responses to robot error. First, we develop NeckNet-18, a 3D facial reconstruction model to map the reactions captured through the chin camera onto facial points and head motion. We then use these facial responses to develop a robot error detection model which outperforms standard methodologies such as using Openface or video data, generalizing well both for within- and across-participant data. Through this work, we argue for expanding human-in-the-loop robot sensing, fostering more seamless integration of robots into diverse human environments, pushing the boundaries of social cue detection and opening new avenues for adaptable and sustainable robotics.

## Repository Structure

The repository follows the following structure:

1. stimulus_dataset_information.xlsx - database of videos used in the Stimulus Dataset
2. code_neckface_training - code used to preprocess NeckfaceIR and train NeckNet-18
3. code_preprocessing - code used to process the following datasets for training: NeckfaceIR, RGBData, NeckData and OpenData
4. code_timeseries - code used to train timeseries models on NeckData and OpenData to detect reactions to Error
5. code_rgb - code used to train CNN pretrained models on NeckfaceIR and RGBData to detect reactions to Error


## Reference this work 

TBD