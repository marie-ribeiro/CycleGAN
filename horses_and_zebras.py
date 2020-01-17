#!/usr/bin/env python
# coding=utf-8
import argparse
import itertools
import os
from random import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data 
import torchvision.datasets as dset 
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np 
import functools
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML
from numpy import asarray
from numpy.random import randint
from random import seed
#from torch.optim.lr_scheduler import LambdaLR


#note The model is designed to take and generate color images with the size 256×256 pixels.
#The architecture is comprised of four models, two discriminator models, and two generator models.
#Two discriminator models are used, one for Domain-A (horses) and one for Domain-B (zebras).
#(NOT YET IMPLEMENTED)** The discriminator uses a PatchGAN model which has the advantage of being able to apply to images of different sizes
# (NOT YET IMPLEMENTED)**It is designed so that each output prediction of the model maps to a 70×70 square or patch of the input image.

datarootHorse = "/db/ezymgr/horseImages" #"/home/marie/Documents/pytorchPractice/horseImages"
datarootZebra = "/db/ezymgr/zebraImages" #"/home/marie/Documents/pytorchPractice/zebraImages" 

image_size = 256#128
batch_size = 1 #works with 3 #20
workers = 1 #shld be 2 
ngpu =  1 #1 #number of GPUs available. Use 0 for CPU mode - currently made 0 in trying to fix 'OOM' errors
num_epochs = 100 #10
currEpoch = 0
decayEpoch = 10 #epoch to begin linearly decaying the learning rate to zero

#create the dataset
datasetHorse = dset.ImageFolder(root=datarootHorse, transform=transforms.Compose([
    transforms.Resize(image_size), transforms.CenterCrop(image_size),
    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5)),
]))
datasetZebra = dset.ImageFolder(root=datarootZebra, transform=transforms.Compose([
    transforms.Resize(image_size), transforms.CenterCrop(image_size),
    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5)),
]))


#create the dataloader
dataloaderHorse = torch.utils.data.DataLoader(datasetHorse, batch_size=batch_size, shuffle = True,
num_workers=workers) #provides an iterable over a given dataset
dataloaderZebra = torch.utils.data.DataLoader(datasetZebra, batch_size=batch_size, shuffle = True,
num_workers=workers)
#dataloader for random 'test' samples #NOT IN USE
dataloaderHorseTest = torch.utils.data.DataLoader(datasetHorse, batch_size=batch_size, shuffle = True,
num_workers=workers) #provides an iterable over a given dataset
dataloaderZebraTest = torch.utils.data.DataLoader(datasetZebra, batch_size=batch_size, shuffle = True,
num_workers=workers)

#Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")

#real_batch = next(iter(dataloaderHorse))
#plt.figure(figsize=(8,8))
#plt.axis("off")
#plt.title("Training Images")
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
#padding=2, normalize=True).cpu(),(1,2,0))) ############################change to save instead of show
#plt.show()

print(len(datasetHorse))
print(len(datasetZebra))

#implementation
#custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__ #get the classname
    if classname.find('Conv') != -1: #-1 returned if not found
        nn.init.normal_(m.weight.data, 0.0, 0.02) #torch.nn
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) #Fills input Tensor with values drawn from the normal distribution: tensor, mean and standard dev
        nn.init.constant_(m.bias.data, 0) # fills tensor with value 0


#DISCRIMINATOR code
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.conv2d_layer1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)#channelsIn, channelOut, kernelSize, stride, padding
        self.leakyR = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        self.conv2d_layer2 = nn.Conv2d(64, 128, 4, 2, 1, bias =False)
        self.instNorm_layer2 = nn.InstanceNorm2d(128)
        self.leakyR = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        self.conv2d_layer3 = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
        self.instNorm_layer3 = nn.InstanceNorm2d(256)
        self.leakyR = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        self.conv2d_layer4 = nn.Conv2d(256, 512 , 4, 1, (1,2), bias = False)
        self.instNorm_layer4 = nn.InstanceNorm2d(512)
        self.leakyR = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        self.conv2d_layer5 = nn.Conv2d(512,1,4,1,(2,1))

    def forward(self, x):
        x = self.leakyR(self.conv2d_layer1(x))
        x = self.leakyR(self.instNorm_layer2(self.conv2d_layer2(x)))
        x = self.leakyR(self.instNorm_layer3(self.conv2d_layer3(x)))
        x = self.leakyR(self.instNorm_layer4(self.conv2d_layer4(x)))
        x = self.conv2d_layer5(x)
        return x

#Create the discriminator
netDa = Discriminator(ngpu).to(device)
netDb = Discriminator(ngpu).to(device)
#Handle multi-gpu if desired
if(device.type == 'cuda') and (ngpu>1):
    netDa = nn.DataParallel(netDa, list(range(ngpu)))
    netDb = nn.DataParallel(netDb, list(range(ngpu)))
    
#Apply the weights_init function to randomly initialize all weights to
#mean=0 , stdev =0.2.
netDa.apply(weights_init) #apply weight initialisation to ALL layers in model
netDb.apply(weights_init)

#Print the model
print(netDa)
print(netDb)

def conv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0, norm=nn.InstanceNorm2d, relu=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False), 
        norm(out_dim),
        relu()
    )


class ResiduleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResiduleBlock, self).__init__()
        conv_bn_relu = conv_norm_act
        self.ls = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_dim, out_dim, 3, 1, 0, bias=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_dim, out_dim, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(out_dim)
        )
    def forward(self, x):
        return x + self.ls(x) #get returned model to initial model
        

#GENERATOR model
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.ref_pad = nn.ReflectionPad2d(3)
        self.conv2d_layer1 = nn.Conv2d(3, 64, 7, 1, 0, bias = "False")
        self.instNorm = nn.InstanceNorm2d(64)
        self.relu1 = nn.ReLU(True)
        #layer 2
        self.conv2d_layer2 = nn.Conv2d(64, 128, 3, 2, 1, bias ="False")
        self.instNorm_layer2 = nn.InstanceNorm2d(128)
        self.relu_layer2 = nn.ReLU(True)
        #;ayer 3
        self.conv2d_layer3= nn.Conv2d(128, 256, 3, 2, 1, bias="False")
        self.instNorm_layer3 = nn.InstanceNorm2d(256)
        self.relu_layer3 = nn.ReLU(True)
        #residule block time
        self.resBlock1 = ResiduleBlock(256,256)
        self.resBlock2 = ResiduleBlock(256,256)
        self.resBlock3 = ResiduleBlock(256,256)
        self.resBlock4 = ResiduleBlock(256,256)
        self.resBlock5 = ResiduleBlock(256,256)
        self.resBlock6 = ResiduleBlock(256,256)
        self.resBlock7 = ResiduleBlock(256,256)
        self.resBlock8 = ResiduleBlock(256,256)
        self.resBlock9 = ResiduleBlock(256,256)
        #more layers
        self.conv2d_layer4 = nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding = 1, bias="False")
        self.instNorm_layer4 = nn.InstanceNorm2d(128)
        self.relu_layer4 = nn.ReLU(True)
        #layer 5
        self.conv2d_layer5 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding = 1, bias="False")
        self.instNorm_layer5 = nn.InstanceNorm2d(64)
        self.relu_layer5 = nn.ReLU(True)
        #final
        self.ref_pad6 = nn.ReflectionPad2d(3)
        self.conv2d_layer6 = nn.Conv2d(64, 3, 7, 1, 0)
        self.tanH = nn.Tanh()

    def forward(self, x):
        #return self.ls(x)
        x = F.relu(self.instNorm(self.conv2d_layer1((self.ref_pad(x)))))
        x = F.relu(self.instNorm_layer2(self.conv2d_layer2(x)))
        x = F.relu(self.instNorm_layer3(self.conv2d_layer3(x)))
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
        x = self.resBlock5(x)
        x = self.resBlock6(x)
        x = self.resBlock7(x)
        x = self.resBlock8(x)
        x = self.resBlock9(x)
        x = F.relu(self.instNorm_layer4(self.conv2d_layer4(x)))
        x = F.relu(self.instNorm_layer5(self.conv2d_layer5(x)))
        x = torch.tanh(self.conv2d_layer6(self.ref_pad6(x)))
        return x

#Create the generator
netGa = Generator(ngpu).to(device)
netGb = Generator(ngpu).to(device)
#handle multi-gpu if desired
if(device.type == 'cuda') and (ngpu > 1):
    netGa == nn.DataParallel(netGa, list(range(ngpu)))
    netGb == nn.DataParallel(netGb, list(range(ngpu)))
#Apply the weights_init function to randomly initialize all weights
#to mean=0, stdev=0.2
netGa.apply(weights_init)
netGb.apply(weights_init)

#print the model
print(netGa)

#Loss Function and Optimizers
#Initialize MSLoss Function (binary cross entropy loss)
MSE = nn.MSELoss()
L1 = nn.L1Loss()

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

#Setup Adam optimizers for both G and D
optimizerDa = optim.Adam(netDa.parameters(), lr=0.0002, betas =(0.5, 0.999))
optimizerDb = optim.Adam(netDb.parameters(), lr=0.0002, betas =(0.5, 0.999))
#optimizerGa = optim.Adam(netGa.parameters(), lr=0.0002, betas =(0.5, 0.999))
#optimizerGb = optim.Adam(netGb.parameters(), lr=0.0002, betas =(0.5, 0.999))
optimizerG = torch.optim.Adam(itertools.chain(netGa.parameters(), netGb.parameters()),lr=0.0002, betas=(0.5, 0.999))  
#Learning Rate Schedulers
lr_schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=LambdaLR(num_epochs, currEpoch, decayEpoch).step)
lr_schedulerDa = torch.optim.lr_scheduler.LambdaLR(optimizerDa, lr_lambda=LambdaLR(num_epochs, currEpoch, decayEpoch).step)
lr_schedulerDb = torch.optim.lr_scheduler.LambdaLR(optimizerDb, lr_lambda=LambdaLR(num_epochs, currEpoch, decayEpoch).step)

#
#update image pool for fake images
seed(1)
def update_image_pool(pool, images, max_size = 50):
    #selected = list()
    for image in images:
        if len(pool) < max_size:
            #stock pool
            pool.append(image)
            selected = image #selected.append(image)
        elif random() < 0.5:
            #use image but don't add to pool
            selected = image #selected.append(image)
        else:
            #replace an existing image and use the replaced image
            ix = randint(0, len(pool))
            selected = pool[ix]#selected.append(pool[ix])
            pool[ix] = image
    return selected
    
a_fake_pool, b_fake_pool = list(), list()

#directory to store output
if not os.path.exists('/home/ezymgr/outputA'): #('/home/marie/Documents/pytorchPractice/outputA'):
    os.makedirs('/home/ezymgr/outputA', exist_ok=True) #('/home/marie/Documents/pytorchPractice/outputA', exist_ok=True)
if not os.path.exists('/home/ezymgr/outputB'): #('/home/marie/Documents/pytorchPractice/outputB'): 
    os.makedirs('/home/ezymgr/outputB', exist_ok=True) #('/home/marie/Documents/pytorchPractice/outputB',exist_ok=True) 

#create if not exist, a file to write output to
outputFile = open("outputCycleGAN.txt","w+")#('/home/marie/Documents/pytorchPractice/outputCycleGAN.txt', 'w+')
outputFileDLOSS = open("outputDLoss.txt","w+")
outputFileGLOSS = open("outputGLoss.txt", "w+")
outputFileEpochs = open("outputEpochs.txt", "w+")
# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
#
G_losses = []
D_losses = []

def run():
    #img_list_horses_fake = []
    #img_list_zebras_fake = []
    epochs = 0
    steps = 0
    print ("starting training loop...")
    for epoch in range (currEpoch, num_epochs):#num_epochs
        #for each batch in dataloader
        for i, (horses_real, zebra_real) in enumerate(zip(dataloaderHorse,dataloaderZebra)):
            #print("epoch, iteration"+ str(epochs)+" "+ str(steps+1))
            n_epochs, n_batch = num_epochs,1
            bat_per_epoch = int(len(horses_real[0].to(device))/n_batch)
            n_steps = bat_per_epoch * n_epochs
            steps = steps+1
            
            h_real = horses_real[0].to(device)
            z_real = zebra_real[0].to(device)
            #Generators H2Z and Z2H
            optimizerG.zero_grad()
            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netGb(z_real)
            loss_identity_B = criterion_identity(same_B, z_real)*5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netGa(h_real)
            loss_identity_A = criterion_identity(same_A, h_real)*5.0
            # GAN loss
            fake_B = netGb(h_real)
            pred_fake = netDb(fake_B)
            target_real = Variable(torch.ones(pred_fake.size())).to(device)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netGa(z_real)
            pred_fake = netDa(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
            # Cycle loss
            recovered_A = netGa(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, h_real)*10.0
            recovered_B = netGb(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, z_real)*10.0
            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            optimizerG.step()

            ###### Discriminator A ######
            optimizerDa.zero_grad()
            # Real loss
            pred_real = netDa(h_real)
            loss_D_real = criterion_GAN(pred_real, target_real)

            #update fakes from pool and get back a single image
            saveFake_A = fake_A
            fake_A = update_image_pool(a_fake_pool, fake_A)
            fake_A = fake_A.unsqueeze(0) #add dimension to go from 3 to epected 4 dim

            # Fake loss
            pred_fake = netDa(fake_A.detach())
            target_fake = Variable(torch.zeros(pred_fake.size())).to(device)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()
            optimizerDa.step()
            ##### Discriminator B ######
            optimizerDb.zero_grad()
            # Real loss
            pred_real = netDb(z_real)
            loss_D_real = criterion_GAN(pred_real, target_real)
            # Fake loss
            saveFake_B= fake_B
            fake_B = update_image_pool(b_fake_pool, fake_B)
            fake_B = fake_B.unsqueeze(0)
            pred_fake = netDb(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
             # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()
            optimizerDb.step()

            ####
            """
            netGa.train() #Horses Generator a
            netGb.train() #zebra Generator b
                                  
            #change to be like what they did!
            h_real = horses_real[0].to(device)#Variable(horses_real[0]) #copies the data to gpu #domain a
            z_real = zebra_real[0].to(device)#Variable(zebra_real[0]) #domain b
            #Train G; Ga takes input from domain b and vice versa
            a_fake = netGa(z_real)
            b_fake = netGb(h_real)

            a_rec = netGa(b_fake)#why?? this step?
            b_rec = netGb(a_fake)

            #Calculate Generator losses
            a_f_dis = netDa(a_fake)
            b_f_dis = netDb(b_fake)
            r_label = Variable(torch.ones(a_f_dis.size())).to(device)
            a_gen_loss = MSE(a_f_dis, r_label)
            b_gen_loss = MSE(b_f_dis, r_label)

            #rec losses
            a_rec_loss = L1(a_rec, h_real)#error as size not equal
            b_rec_loss = L1(b_rec, z_real)

            #g loss
            g_loss = a_gen_loss + b_gen_loss + a_rec_loss * 10.0 + b_rec_loss * 10.0 #why??
            #print("everywhere")
            #backward pass
            netGa.zero_grad()
            netGb.zero_grad()
            g_loss.backward(retain_graph = True) 
            optimizerGa.step()
            optimizerGb.step()

            #leaves
            #update fakes from pool and get back a single image
            a_fake = update_image_pool(a_fake_pool, a_fake)
            b_fake = update_image_pool(b_fake_pool, b_fake)
            a_fake = a_fake.unsqueeze(0) #add dimension to go from 3 to epected 4 dim
            b_fake = b_fake.unsqueeze(0)
            #print(a_fake.size())
            #print(h_real.size())
            #print(type(a_fake))
            #train D
            a_r_dis = netDa(h_real)
            a_f_dis = netDa(a_fake)
            b_r_dis = netDb(z_real)
            b_f_dis = netDb(b_fake)
            r_label = Variable(torch.ones(a_f_dis.size())).to(device)
            f_label = Variable(torch.zeros(a_f_dis.size())).to(device)

            #d loss
            a_d_r_loss = MSE(a_r_dis, r_label)
            a_d_f_loss = MSE(a_f_dis, f_label)
            b_d_r_loss = MSE(b_r_dis, r_label)
            b_d_f_loss = MSE(b_f_dis, f_label)

            a_d_loss = a_d_r_loss + a_d_f_loss
            b_d_loss = b_d_r_loss + b_d_f_loss
            #print("over")
            #backward pass
            netDa.zero_grad()
            netDb.zero_grad()
            a_d_loss.backward(retain_graph = True)
            b_d_loss.backward(retain_graph = True)
            optimizerDa.step()
            optimizerDb.step()
        """
            #summarize performance
            outputFile.write('>%d,loss_G[%.3f],loss_G_identity[%.3f],loss_G_GAN[%.3f],loss_G_cycle[%.3f], loss_D[%.3f]\n' % (i,loss_G,(loss_identity_A + loss_identity_B),(loss_GAN_A2B + loss_GAN_B2A),
                    (loss_cycle_ABA + loss_cycle_BAB),(loss_D_A + loss_D_B)))
           #outputFile.write('>%d, dA_real,fake[%.3f, %.3f] dB_real,fake[%.3f, %.3f] g_a,b[%.3f, %.3f]\n' % (i, a_d_r_loss,a_d_f_loss,b_d_r_loss, b_d_f_loss, a_gen_loss, b_gen_loss))
            if(steps%500 == 0) or ((epochs==num_epochs-1) and (i==len(dataloaderHorse)-1)):
                vutils.save_image(saveFake_A, '/home/ezymgr/outputA/'+str(epoch)+'_'+str(i)+'_horse'+'.png')
                vutils.save_image(saveFake_B, '/home/ezymgr/outputB/'+str(epoch)+'_'+str(i)+'_zebra'+'.png')
                G_losses.append(loss_G.item())
                D_losses.append(loss_D_A.item()+loss_D_B.item())
                #new
                outputFileEpochs.write('%d\n' %epochs)
                outputFileDLOSS.write('%d\n' %(loss_D_A.item()+loss_D_B.item()))
                outputFileGLOSS.write('%d\n' %(loss_G.item()))

                #img_list_horses_fake.append(vutils.make_grid(a_fake, padding=2, normalize=True))#mem allocation err
                #img_list_zebras_fake.append(vutils.make_grid(b_fake, padding=2, normalize=True))
        epochs =epochs+1
        # Update learning rates
        lr_schedulerG.step()
        lr_schedulerDa.step()
        lr_schedulerDb.step()

        #evaluate the model performance every so often
        #plot A->B translation
        #realHorse =iter(dataloaderHorseTest).next()[0].to(device)
        #realZebra = iter(dataloaderZebraTest).next()[0].to(device)
        #netGa.eval()
        #netGb.eval()
        #fakeHorse = 0.5*(netGa(realZebra) + 1.0)#netGa generates horses
        #fakeZebra = 0.5*(netGb(realHorse) + 1.0)#netGb generates zebras
        #save image files
        #vutils.save_image(fakeHorse, 'db/ezymgr/outputA/'+'str(epoch)'+'_'+'str(i)'+'.png') #'/home/marie/Documents/pytorchPractice/outputA/%04d.png' %(i)) 
        #vutils.save_image(fakeZebra, 'db/ezymgr/outputB/'+'str(epoch)'+'_'+'str(i)'+'.png') #'/home/marie/Documents/pytorchPractice/outputB/%04d.png' %(i))

#Loss versus training iteration
#Below is a plot of D and G's losses versus training iterations
plt.figure(figsize=(10,5))
plt.title("Generator Loss During Training")
plt.plot(np.linspace(1,len(G_losses),len(G_losses)),G_losses, label="G")
#plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
#plt.show()
plt.savefig("Loss_During_Training_GLoss.png")

plt.figure(figsize=(10,5))
plt.title("Discriminator Loss During Training")
plt.plot(np.linspace(1,len(D_losses),len(D_losses)),D_losses, label="D")
#plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
#plt.show()
plt.savefig("Loss_During_Training_DLoss.png")

if __name__ == '__main__':
    run()
    #only added as a 'fix' for issues with OOM (out of memory error 1 or more processes killed)
