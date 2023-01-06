Prostate Cancer Detection HRNet Model
#Before running this application, make sure you have Python>=3.8, CUDA 11.6, CUDAnn, PyTorch 1.13, and a pwsh/zsh/cmd

Step 1: Download and extract the (.zip) file.
Step 2: Open the extracted folder and select open in terminal
Step 3: Run 'pip install -r requirements.txt' to install all the required libraries into your computer. This may take a while.
Step 4: Run 'python docs/split.py' to split your data into train, test and val folders. Configure the input and output data directories in the split.py file.
Step 5: To run the User Interface(UI) for our application, run 'python main.py'.
Step 6: Then, open your browser and paste this URL 'http://127.0.0.1:5000/home'
Step 7: Now you can select any pictures from your test folder and click Upload. Our application will run the pretrained HRNet model to predict the output for the picture that you have uploaded. (e.g: Benign, Grade 3, Grade 4, Grade 5)
Step 8: Now for the training part, you can simply run 'python tools/train.py --cfg experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml' to train the model. Since we already trained our model until 100 epochs it will not train further. You can always change the preferred number of epochs in the (.yaml) file. The default number of epochs is 100. If you don't have powerful GPUs, don't mind reducing the batch_size to 8 or 16 in the (.yaml) file.
Step 9: For the validation part, you can run 'python tools/valid.py --cfg experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel output\imagenet\cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100\model_best.pth.tar'. It will show the accuracy of our pretrained model. In our case, we have an accuracy of 94.444%.