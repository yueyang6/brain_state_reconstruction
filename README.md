# brain_state_reconstruction

Yueyang Liu et al 2023 J. Neural Eng. 20 036024
DOI 10.1088/1741-2552/acd871

## Environment
Reommend to create an environment form the env.yml file with the following command.

```
conda env create -f env.yml
```

For Mac users, you might find conflicts when trying the above command.


Please create a Python 3.7 conda environment, and install the following packages.

numpy==1.18.5

scipy==1.4.1

pandas==1.3.5

tqdm==4.64.1

scikit-learn==1.0.2

matplotlib==3.5.3

statsmodels==0.13.5

tensorflow==2.3.0

## Data Generation
In the "simu_data" folder, you can run the file to generate a small dataset of simulation recordings which is approximately 600MB. A larger dataset is used for the paper by setting the step as 0.01/3, which will generate recordings around 10GB.

The recordings are generated following the flow chart. Ljung-Box test is used for hypothesis testing.
![fig2_v2](https://user-images.githubusercontent.com/54312398/208929960-237fdfc5-af98-4744-a924-4a972c87fac0.png)


## Training
By running "custom_loss_bi_noise.py" file, the model will be trained on the training data located in the "simu_data" folder.

Or the trained weights that the paper has used is provided in the "saved_weights" folder.


## Testing
"random_recording_test.py" will randomly generate a recording with six different combinations of $\tau_e$ and $\tau_i$. Both Kalman filter and the LSTM filter will be compared. The generated figure will look like this.
![random_simulation](https://user-images.githubusercontent.com/54312398/207226657-ba39db0e-b0f0-4dcd-b2f9-034dd2c82688.jpg)
