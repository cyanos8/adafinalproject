HOW TO NAVIGATE THIS DIRECTORY

The primary three files in this directory are Read_Video.py, Sapiens_Demo.ipynb and Eulerian_Magnification.py. 
The code in Read_Video (~44 lines) reads in a provided sample video and saves the frames in the data folder

The code in Sapiens_Demo (~182 lines total)
    (1) Downloads necesary checkpoints from huggingface
    (2) Uses code adapted from LearnOpenCV.com by Jaykumaren:
            https://learnopencv.com/sapiens-human-vision-models/
    (3) Implements said code to segment the face-neck area from my input video, and mask all other aspects of the frame
    (4) Saves the masked frames to the processed_data folder

The code in Eulerian_Magnification (~168 lines) implements the Euelerian Magnification Algorithm described in paper on the processed video frames in the processed_data folder.