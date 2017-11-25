Sorry about the ugly coding style...

The unit-level features should be saved with the name of (movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"), where movie_name is the name of the video, swin_start and swin_end are the start and end frame of the window. For example, video_test_0001495.mp4_10809.0_10825.0.npy. The unit-level feature should be extract every 8 frames (the unit size is 16 frames, so there will be 50% overlap.). I upload my denseflow CNN unit features (size 16, overlap 50%) to Google Drive, val set: [https://drive.google.com/file/d/1-6dmY_Uy-H19HxvfK_wUFQCYHmlPzwFx/view?usp=sharing], test set: [https://drive.google.com/file/d/1Qm9lIJQFm5s6hDSB_2k1tj8q2tnabflJ/view?usp=sharing]

Once you have the unit-level features, edit the feature path in main.py, and then just run python main.py. Best model should be trained in about 10000 steps with current training samples.
