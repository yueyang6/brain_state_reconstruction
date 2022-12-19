# brain_state_reconstruction

## Environment
Reommend to create an environment form the env.yml file with the following command.

```
conda env create -f env.yml
```

For Mac users, you might find conflicts when trying the above command.


Please create a Python 3.7 conda environment, and install the following packages.

numpy==1.18.5

sicpy==1.4.1

pandas==1.3.5

tqdm==4.64.1

scikit-learn==1.0.2

matplotlib==3.5.3

statsmodels==0.13.5


## Data Generation
In the "simu_data" folder, you can run the file to generate a small dataset of simulation recordings which is approximately 600MB.

The recordings are generated following the flow chart. Ljung-Box test is used for hypothesis testing.
![fig2](https://user-images.githubusercontent.com/54312398/207741349-56cedea1-4d49-4fff-a022-49e59ec61074.png)


## Training
By running "custom_loss_bi_noise.py" file, the model will be trained on the training data located in the "simu_data" folder.

Or the trained weights same as the paper has used is provided in the "saved_weights" folder.


## Testing
"random_recording_test.py" will randomly generate a recording with six different combinations of $\tau_e$ and $\tau_i$. Both Kalman filter and the LSTM filter will be compared. The generated figure will look like this.
![random_simulation](https://user-images.githubusercontent.com/54312398/207226657-ba39db0e-b0f0-4dcd-b2f9-034dd2c82688.jpg)
