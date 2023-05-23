# NCDL
 
1,NCDL-GD transmits the general data. 
  the codeword coming from the channel codes is proceesed by network coding. Then the codeword after coding can be recovered.
  So, NCDL-GD can transfer general code with network coding and the traditioanl chanael code. the running envirment is keras+tensorflow


2, NCDL-SD can transmit special dataset. 
  in this demo, the datasets are MINIST dataset and Geographic dataset, and they are transmitted through butterfly network where network coding
  are performed. The code for MINIST dataset and Geographic dataset are in two sub-dirs which are NCDL_GD_Geography_dataset and NCDL_SD_for MNIST. 
  The deep leaning frame is pytorch.
