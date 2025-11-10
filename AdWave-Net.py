from networkx import selfloop_edges
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.dwt_modules.DWT_IDWT_layer import *


def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv

class Wavelet_spatial_attention(nn.Module):
    def __init__(self,in_channels, WF_type='WS4'):
        super(Wavelet_spatial_attention,self).__init__()
        
        self.WF_type = WF_type
        #  (N, C, H, W) --> LL, LH, HL, HH (N, C, H/2, W/2)
        self.DWT_down = DWT_2D()     
        
        #  3C --> C 
        c3 = in_channels*3
        c2 = in_channels*2
        self.conv_c3_c = nn.Sequential(nn.Conv2d(c3, in_channels,kernel_size=1, padding=0),
                                     nn.ReLU(inplace=True)) 
        
        self.conv_c2_c = nn.Sequential(nn.Conv2d(c2, in_channels,kernel_size=1, padding=0),
                                     nn.ReLU(inplace=True))
        
        
        self.x2conv_c3_c = nn.Sequential(           
            nn.Conv2d(c3, c2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, in_channels, kernel_size=1,padding= 0),
            nn.Sigmoid()
        )
        self.x2conv_c2_c = nn.Sequential(           
            nn.Conv2d(c2, in_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1,padding= 0),
            nn.Sigmoid()
        )
        self.x2conv_c_c = nn.Sequential(           
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1,padding= 0),
            nn.Sigmoid()
        )
        
        self.x1conv_c_1 = nn.Conv2d(in_channels,1, kernel_size=1, stride=1, padding=0)
        
        self.AvgPool = nn.AvgPool2d(2,stride=2)
        self.MaxPool = nn.MaxPool2d(2,stride=2)   
 
    
    
    
    def WF4(self,x):
        """
        input: x    (B,C,H,W)
        F: add(LL, FC(LH, HL, HH ))       (B,C,H/2,W/2  each), ours
        att_map:    (B,C,H/2,W/2)   
        output: Maxpool(x).multipy(att_map)  B,C,H/2,W/2 
        """
        LL,LH, HL, HH = self.DWT_down(x)
        F_dwt = LL + self.conv_c3_c(torch.cat([LH,HL,HH],dim=1))
        att_map = self.x2conv_c_c(F_dwt)
        return  att_map
    
    def WF5(self,x):
        """
        input: x    (B,C,H,W)
        F: cat(LL, FC(LH, HL, HH ))       (B,C,H/2,W/2  each), ours
        att_map:    (B,C,H/2,W/2)   
        output: Maxpool(x).multipy(att_map)  B,C,H/2,W/2 
        """
        LL,LH, HL, HH = self.DWT_down(x)
        F_h = self.conv_c3_c(torch.cat([LH,HL,HH],dim=1))
        att_map = self.x2conv_c2_c(torch.cat([LL,F_h],dim=1))
        return  att_map
    
 
                
    def forward(self,x):
        if self.WF_type =='WF4':
            return self.WF4(x)
        
        elif self.WF_type =='WF5':
            return self.WF5(x)
        
       
        else:
            raise TypeError
        
class HW_DWT_module(nn.Module):
    def  __init__(self,in_channels,Down_type,Fusion_type,WF_type='WF4'):
        super(HW_DWT_module,self).__init__()  

        self.Down_type = Down_type
        self.Fusion_type = Fusion_type
        self.WF_type = WF_type
        
        self.DWT_down = DWT_2D()
        self.conv_c3_c = nn.Sequential(nn.Conv2d(in_channels*3, in_channels,kernel_size=1, padding=0),
                                     nn.ReLU(inplace=True))
        self.x2conv_c_c = nn.Sequential(           
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1,padding= 0),
            nn.Sigmoid()
        )
        self.x2conv_c2_c = nn.Sequential(           
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1,padding= 0),
            nn.Sigmoid()
        )
                 
        self.MaxPool = nn.MaxPool2d(2,stride=2)    
        self.AvgPool = nn.AvgPool2d(2,stride=2)
        
        self.x1conv_c_2c = nn.Conv2d(in_channels,in_channels*2, kernel_size=1, stride=1, padding=0)        
        self.x1conv_2c_2c = nn.Conv2d(in_channels*2,in_channels*2, kernel_size=1, stride=1, padding=0)
        self.x1conv_4c_2c = nn.Conv2d(in_channels*4,in_channels*2, kernel_size=1, stride=1, padding=0)
        
        
    def WF(self,x):
        """
        input: x    (B,C,H,W)
        F: add(LL, FC(LH, HL, HH ))       (B,C,H/2,W/2  each), ours
        att_map:    (B,C,H/2,W/2)   
        output: Maxpool(x).multipy(att_map)  B,C,H/2,W/2 
        """
        LL,LH, HL, HH = self.DWT_down(x) 
        if self.WF_type == "WF4":     
            F_h = self.conv_c3_c(torch.cat([LH,HL,HH],dim=1))            
            att_map = self.x2conv_c_c(LL + F_h)
            
        elif self.WF_type == "WF5": 
            F_h = self.conv_c3_c(torch.cat([LH,HL,HH],dim=1))
            att_map = self.x2conv_c2_c(torch.cat([LL,F_h],dim=1))
            
        else:
            raise TypeError
        
        return  att_map, LL,LH, HL, HH, F_h
    
    
    
    def Down1(self,x):
        att_map, LL, _, _, _, _ = self.WF(x)
        return att_map, LL
    
    def Down2(self,x):
        att_map, LL, _, HL, _, _ = self.WF(x)
        return att_map, LL+HL
    
    def Down3(self,x):
        att_map, LL, _, _, _, F_h = self.WF(x)
        return  att_map, LL+F_h
    
    def Down(self,x):
        if self.Down_type =='0':
            return self.Down0(x)
        elif self.Down_type =='1':
            return self.Down1(x)
        elif self.Down_type =='2':
            return self.Down2(x)
        elif self.Down_type =='3':
            return self.Down3(x)
        else:
            raise TypeError
    
    def Fusion(self, x_1,x_dwt):
        x_dwt = self.x1conv_c_2c(x_dwt)
        
        if self.Fusion_type =='0':
            return self.x1conv_2c_2c(torch.add(x_1,x_dwt))
        elif self.Fusion_type =='1':
            return self.x1conv_4c_2c(torch.cat([x_1, x_dwt],dim=1))
        elif self.Fusion_type =='2':
            return self.x1conv_2c_2c(torch.multiply(x_1, x_dwt))
        else:
            raise TypeError


class AdWave(nn.Module):
    def __init__(self,in_channels,num_classes,Down_type=3,Fusion_type=1,WF_type='WF5'):
        super(AdWave,self).__init__()   
        
        Down_type,Fusion_type = str(Down_type),str(Fusion_type)
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
          
        self.start_conv = x2conv(in_channels, 64)   
        
        self.DWT1 = HW_DWT_module(64,Down_type,Fusion_type,WF_type)
        self.conv1 = x2conv(64, 128) 
         
        self.DWT2 = HW_DWT_module(128,Down_type,Fusion_type,WF_type)
        self.conv2 = x2conv(128, 256)  
        
        self.DWT3 = HW_DWT_module(256,Down_type,Fusion_type,WF_type)
        self.conv3 = x2conv(256, 512)  
        
        self.DWT4 = HW_DWT_module(512,Down_type,Fusion_type,WF_type)
        self.conv4 = x2conv(512, 1024)  
        
        self.middle_conv = x2conv(1024, 1024)  
        
        self.uppool4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)       
        self.dec_conv4 = x2conv(1024, 512)       
        
        self.uppool3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)        
        self.dec_conv3 = x2conv(512, 256) 
        
        self.uppool2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)        
        self.dec_conv2 = x2conv(256, 128) 
        
        self.uppool1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)        
        self.dec_conv1 = x2conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self,x):
        # encoding path
        x1 = self.start_conv(x)  #B,64,512,512
        
        att1,copy1 = self.DWT1.Down(x1)
        x2 = torch.mul(self.Maxpool(x1), att1)  #B, 64, 256,256         
        x2 = self.conv1(x2) #B, 128, 256,256
        
        att2,copy2 = self.DWT2.Down(x2)
        x3 = torch.mul(self.Maxpool(x2), att2)           
        x3 = self.conv2(x3) #B, 512, 128,128
        
        att3,copy3 = self.DWT3.Down(x3)
        x4 = torch.mul(self.Maxpool(x3), att3)           
        x4 = self.conv3(x4)  #B,1024,64,64
        
        att4,_ = self.DWT4.Down(x4)
        x5 = torch.mul(self.Maxpool(x4), att4)   
        x5 = self.conv4(x5)  #B,1024,32,32
                
        x5 = self.middle_conv(x5) #B,1024,32,32
        
                
        d4 = self.uppool4(x5) #d4: B, 512, 64, 64
        x4 = self.DWT3.Fusion(x4,copy3) # input: copy3:256, x4:512; out:512
        d4 = torch.cat((x4,d4), dim=1) # B, 1024, 128,128
        d4 = self.dec_conv4(d4) #B,512,64,64
        
        d3 = self.uppool3(d4) #B,256,128,128
        x3 = self.DWT2.Fusion(x3,copy2) #input:256,128;out:256
        d3 = torch.cat((x3,d3), dim=1) #256,256，--》512
        d3 = self.dec_conv3(d3) #256
        
        d2 = self.uppool2(d3)
        x2 = self.DWT1.Fusion(x2,copy1)
        d2 = torch.cat((x2,d2),dim=1)
        d2 = self.dec_conv2(d2)
        
        d1 = self.uppool1(d2)
        d1 = torch.cat((x1,d1),dim=1)
        d1 = self.dec_conv1(d1)
               
        d = self.final_conv(d1)      

        return d



if __name__ == '__main__':
    
  
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,7'
    img= 'oct2.png'
    

    
    x = torch.randn([1,3,512,512]).cuda()
    model = AdWave(in_channels=3, num_classes=4,
                Down_type="1",
                Fusion_type="0").cuda()
    
    
    print(model(x).shape)
    
  
        
